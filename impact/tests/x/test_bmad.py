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

# Bmad element types with no ImpactX mapping yet (from_tao raises for these).
_unsupported = pytest.mark.xfail(reason="Element type not yet supported", strict=False)
lattice_markers = {
    "elements.bmad": _unsupported,
    "lcavity.bmad": _unsupported,
    "lcavity_rf.bmad": _unsupported,
    "wiggler.bmad": _unsupported,
    "decapole.bmad": _unsupported,
    "kickers.bmad": _unsupported,
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
    "solenoid.bmad": [
        pytest.mark.xfail(reason="Solenoid x-y coupling mismatch (ks scaling)")
    ],
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
