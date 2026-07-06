from __future__ import annotations

import pathlib

import numpy as np
import pytest
from beamphysics import ParticleGroup, single_particle
from beamphysics.units import mec2
from pytao import SubprocessTao as Tao

from ...x import ImpactX, ImpactXInput
from .conftest import bmad_files

lattice_root = bmad_files

# Bmad element types with no ImpactX mapping (from_tao raises for these).
# `wiggler` has no ImpactX element; `elements.bmad` contains an `e_gun`.
_unsupported = pytest.mark.xfail(reason="Element type not yet supported", strict=False)
lattice_markers = {
    "elements.bmad": _unsupported,
    "wiggler.bmad": _unsupported,
}

lattices = pytest.mark.parametrize(
    "lattice",
    [
        pytest.param(fn, id=fn.name, marks=lattice_markers.get(fn.name, []))
        for fn in sorted(lattice_root.glob("*.bmad"))
    ],
)

# Single-element lattices whose on-axis trajectory should match Tao,
# with per-lattice pytest marks.
comparison_lattices: dict[str, list] = {
    "drift.bmad": [],
    "quad.bmad": [],
    "dipole.bmad": [],
    "sextupole.bmad": [],
    "octupole.bmad": [],
    "decapole.bmad": [],
    "solenoid.bmad": [],
    "kickers.bmad": [],
    # lcavity/lcavity_rf convert (see test_from_tao) but are omitted here: a
    # transverse single-particle orbit is insensitive to the longitudinal RF
    # field, so compare_sxy cannot validate RF physics.  Calibrating the RF
    # field profile and adding a longitudinal (energy) check is future work.
}


@lattices
def test_from_tao(lattice: pathlib.Path) -> None:
    """Smoke test: convert each lattice's live Tao model to an ImpactXInput."""
    with Tao(lattice_file=str(lattice), noplot=True) as tao:
        input = ImpactXInput.from_tao(tao)
    assert input.lattice
    print(input)


def set_initial_particles(tao: Tao, P0: ParticleGroup, path: pathlib.Path) -> None:
    fn = path / "initial_particles.h5"
    P0.write(str(fn))
    tao.cmds(
        [
            f"set beam_init position_file = {fn}",
            f"set beam_init n_particle = {len(P0)}",
            f"set beam_init bunch_charge = {P0.charge}",
            "set beam_init saved_at = *",
            "set global track_type = single",
            "set global track_type = beam",
        ]
    )


@pytest.mark.parametrize(
    "lattice",
    [
        pytest.param(lattice_root / name, id=name, marks=marks)
        for name, marks in comparison_lattices.items()
    ],
)
def test_compare_sxy(
    tmp_path: pathlib.Path,
    lattice: pathlib.Path,
) -> None:
    """Compare the mean transverse orbit through one element against Tao."""
    energy = 10e6
    pz = np.sqrt(energy**2 - mec2**2)
    P0 = single_particle(x=1e-3, pz=pz, species="electron")

    with Tao(lattice_file=str(lattice), noplot=True) as tao:
        tao.cmd("set beam comb_ds_save = 0.02")
        set_initial_particles(tao, P0, tmp_path)
        input = ImpactXInput.from_tao(tao)
        s_tao = np.array(tao.bunch_comb("s"))
        x_tao = np.array(tao.bunch_comb("x"))
        y_tao = np.array(tao.bunch_comb("y"))

    for ele in input.lattice:
        if hasattr(ele, "nslice"):
            ele.nslice = max(ele.nslice, 20)

    I = ImpactX(input, workdir=tmp_path)
    output = I.run()

    s = output.stats["s"]
    x = output.stats["mean_x"]
    y = output.stats["mean_y"]
    x_tao_interp = np.interp(s, s_tao, x_tao)
    y_tao_interp = np.interp(s, s_tao, y_tao)

    atol = 1e-4
    np.testing.assert_allclose(x, x_tao_interp, atol=atol, err_msg="X differs")
    np.testing.assert_allclose(y, y_tao_interp, atol=atol, err_msg="Y differs")


# --- CSR ---------------------------------------------------------------------
#
# CSR is a collective effect, so the single-particle compare_sxy harness cannot
# exercise it, and a cross-code (Bmad vs ImpactX) CSR benchmark is out of scope.
# These tests verify that (1) from_tao propagates Bmad's global CSR flag, and
# (2) ImpactX actually applies CSR on the converted lattice, producing the
# expected energy-spread growth in a bend.

# (lattice, whether Bmad enables the global CSR flag).  csr_zeuthen.bmad has
# ``csr_and_space_charge_on`` commented out, so CSR is off there.
csr_lattices = [("csr_bench.bmad", True), ("csr_zeuthen.bmad", False)]


def _gaussian_bunch(kin_energy_MeV: float, n: int = 2000) -> ParticleGroup:
    """A small Gaussian electron bunch at the given reference kinetic energy."""
    rng = np.random.default_rng(0)
    e_tot = kin_energy_MeV * 1e6 + mec2
    pz0 = np.sqrt(e_tot**2 - mec2**2)
    data = {
        "x": rng.normal(0, 50e-6, n),
        "y": rng.normal(0, 50e-6, n),
        "z": rng.normal(0, 200e-6, n),
        "px": rng.normal(0, 1e3, n),
        "py": rng.normal(0, 1e3, n),
        "pz": pz0 + rng.normal(0, 1e3, n),
        "t": np.zeros(n),
        "weight": np.full(n, 1e-9 / n),
        "status": np.ones(n),
        "species": "electron",
    }
    return ParticleGroup(data=data)


@pytest.mark.parametrize(
    ("lattice", "expected_csr"),
    [pytest.param(name, csr, id=name) for name, csr in csr_lattices],
)
def test_csr_flag_from_tao(lattice: str, expected_csr: bool) -> None:
    """Bmad ``csr_and_space_charge_on`` becomes ``ImpactXInput.csr``."""
    with Tao(lattice_file=str(lattice_root / lattice), noplot=True) as tao:
        input = ImpactXInput.from_tao(tao)
    assert input.csr is expected_csr


@pytest.mark.slow
def test_csr_applied(tmp_path: pathlib.Path) -> None:
    """ImpactX applies CSR on the converted lattice (collective sanity check).

    A bend with CSR enabled must grow the beam energy spread far beyond the
    CSR-off case; this confirms the flag/mesh plumbing rather than benchmarking
    the CSR model against Bmad.
    """
    with Tao(lattice_file=str(lattice_root / "csr_bench.bmad"), noplot=True) as tao:
        input = ImpactXInput.from_tao(tao)

    input.initial_particles = _gaussian_bunch(input.kin_energy_MeV)
    input.bunch_charge_C = input.initial_particles.charge
    for ele in input.lattice:
        if hasattr(ele, "nslice"):
            ele.nslice = max(ele.nslice, 40)

    input.csr = False
    off = ImpactX(input, workdir=tmp_path).run()

    input.csr = True
    on = ImpactX(input, workdir=tmp_path).run()

    assert on.stats["sigma_pt"][-1] > 10.0 * off.stats["sigma_pt"][-1]
