from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from beamphysics import ParticleGroup, single_particle
from beamphysics.units import c_light, mec2
from pytao import SubprocessTao as Tao

from ...x import ImpactX, ImpactXInput
from .conftest import bmad_files, test_artifacts

lattice_root = bmad_files


def _save_trajectory_comparison(
    name: str,
    s: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    s_tao: np.ndarray,
    x_tao: np.ndarray,
    y_tao: np.ndarray,
    atol: float,
) -> None:
    """Save a Bmad-vs-ImpactX mean-orbit comparison figure to ``artifacts/``."""
    x_i = np.interp(s, s_tao, x_tao)
    y_i = np.interp(s, s_tao, y_tao)
    passed = np.allclose(x, x_i, atol=atol) and np.allclose(y, y_i, atol=atol)

    fig, (axx, axy) = plt.subplots(2, 1, sharex=True, figsize=(9, 6))
    fig.suptitle(f"{name} — {'PASS' if passed else 'FAIL'} (atol={atol:g})")
    for ax, comp, tao, imp in (
        (axx, "x", x_tao, x),
        (axy, "y", y_tao, y),
    ):
        ax.plot(s_tao, tao, "--", color="C0", label="Bmad/Tao")
        ax.plot(s, imp, "-", color="C3", alpha=0.8, label="ImpactX")
        ax.set_ylabel(rf"mean ${comp}$ [m]")
        ax.grid(alpha=0.3)
    # residual on a twin axis of the x panel
    axr = axx.twinx()
    axr.plot(s, x - x_i, color="0.6", lw=0.8)
    axr.set_ylabel("Δx (ImpactX−Tao) [m]", color="0.5")
    axy.set_xlabel("s [m]")
    axx.legend(loc="best", fontsize=8)

    test_artifacts.mkdir(parents=True, exist_ok=True)
    fig.savefig(test_artifacts / f"{name}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def _save_bunch_comparison(
    name: str,
    groups: dict[str, ParticleGroup],
    projections: tuple[tuple[str, str], ...] = (
        ("x", "px"),
        ("y", "py"),
        ("t", "energy"),
    ),
) -> None:
    """Save a phase-space comparison grid of several bunches to ``artifacts/``.

    Rows are phase-space projections, columns are the named bunches (e.g.
    ``{"Bmad": P_tao, "ImpactX": P_ix}``).  Each panel is a 2D density of the
    corresponding :class:`ParticleGroup` quantities; axis limits are shared
    across a row so the bunches can be compared directly.
    """
    labels = list(groups)
    nrow, ncol = len(projections), len(labels)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow), squeeze=False)
    fig.suptitle(name)
    for r, (kx, ky) in enumerate(projections):
        # shared limits across the row (all bunches, same projection)
        xs = np.concatenate([np.asarray(g[kx], dtype=float) for g in groups.values()])
        ys = np.concatenate([np.asarray(g[ky], dtype=float) for g in groups.values()])
        xr = (xs.min(), xs.max()) if xs.min() != xs.max() else None
        yr = (ys.min(), ys.max()) if ys.min() != ys.max() else None
        for c, label in enumerate(labels):
            ax = axes[r][c]
            g = groups[label]
            ax.hist2d(
                np.asarray(g[kx], dtype=float),
                np.asarray(g[ky], dtype=float),
                bins=max(20, int((len(g) / 4) ** 0.5)),
                range=[xr, yr] if xr and yr else None,
                cmap="viridis",
            )
            ax.set_xlabel(kx)
            ax.set_ylabel(ky)
            if r == 0:
                ax.set_title(label)
    test_artifacts.mkdir(parents=True, exist_ok=True)
    fig.savefig(test_artifacts / f"{name}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


# Bmad element types with no ImpactX mapping (from_tao raises for these).
# `elements.bmad` contains an `e_gun`.
_unsupported = pytest.mark.xfail(reason="Element type not yet supported", strict=False)
lattice_markers = {
    "elements.bmad": _unsupported,
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
    "wiggler.bmad": [],
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
    request: pytest.FixtureRequest,
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
    _save_trajectory_comparison(
        request.node.name.replace("/", "_"),
        s,
        x,
        y,
        s_tao,
        x_tao,
        y_tao,
        atol,
    )
    np.testing.assert_allclose(x, x_tao_interp, atol=atol, err_msg="X differs")
    np.testing.assert_allclose(y, y_tao_interp, atol=atol, err_msg="Y differs")


# --- CSR ---------------------------------------------------------------------
#
# CSR is a collective effect, so the single-particle compare_sxy harness cannot
# exercise it: it needs a real bunch.  test_csr_applied is a genuine cross-code
# benchmark -- the same bunch is tracked through a bend with CSR in both Bmad and
# ImpactX and the output beam statistics are compared.
#
# Statistics are taken from ImpactX's authoritative reduced-beam diagnostics
# (not the monitor read-back, whose lab-frame reconstruction currently scales
# momentum-spread quantities by cos(bend angle) -- see
# impactx_monitor_to_particle_group).  ImpactX's own emittance matches Bmad
# exactly, confirming that discrepancy is a read-back artifact only.

# (lattice, whether Bmad enables the global CSR flag).  csr_zeuthen.bmad has
# ``csr_and_space_charge_on`` commented out, so CSR is off there.
csr_lattices = [("csr_bench.bmad", True), ("csr_zeuthen.bmad", False)]

# Macroparticles for the cross-code CSR benchmark.  Raise (e.g. 1_000_000) for a
# high-statistics run; the default keeps the (Tao + ImpactX) run tractable.
CSR_N_PARTICLES = 100_000


def _gaussian_bunch(
    kin_energy_MeV: float,
    n: int,
    *,
    charge_C: float = 100e-12,
    sigma_xy: float = 50e-6,
    sigma_length: float = 100e-6,
    rel_energy_spread: float = 1e-4,
    divergence: float = 1e-5,
) -> ParticleGroup:
    """
    A Gaussian electron bunch at fixed s (spread in time, ``z=0``).

    The longitudinal extent lives in ``t`` (the accelerator convention Bmad's
    CSR binning expects); ``sigma_length`` is the RMS bunch length ``c*sigma_t``.
    """
    rng = np.random.default_rng(0)
    pz0 = np.sqrt((kin_energy_MeV * 1e6 + mec2) ** 2 - mec2**2)
    return ParticleGroup(
        data={
            "x": rng.normal(0, sigma_xy, n),
            "y": rng.normal(0, sigma_xy, n),
            "z": np.zeros(n),
            "px": rng.normal(0, pz0 * divergence, n),
            "py": rng.normal(0, pz0 * divergence, n),
            "pz": pz0 * (1 + rng.normal(0, rel_energy_spread, n)),
            "t": rng.normal(0, sigma_length / c_light, n),
            "weight": np.full(n, charge_C / n),
            "status": np.ones(n, dtype=int),
            "species": "electron",
        }
    )


def _track_tao_bunch(
    lattice: pathlib.Path,
    P: ParticleGroup,
    path: pathlib.Path,
    *,
    csr: bool,
    n_bin: int = 40,
    ds_step: float = 0.05,
) -> tuple[ParticleGroup, ImpactXInput]:
    """
    Track ``P`` through ``lattice`` in Tao and return (final bunch, ImpactXInput).

    The returned :class:`ImpactXInput` carries Tao's exported *initial* bunch, so
    ImpactX starts from an identical distribution.  CSR requires Bmad's
    ``space_charge_com`` binning to be configured.
    """
    fn = path / "csr_in.h5"
    P.write(str(fn))
    with Tao(lattice_file=str(lattice), noplot=True) as tao:
        cmds = [
            f"set beam_init position_file = {fn}",
            f"set beam_init n_particle = {len(P)}",
            f"set beam_init bunch_charge = {P.charge}",
            "set beam_init saved_at = *",
        ]
        if csr:
            cmds += [
                f"set space_charge_com n_bin = {n_bin}",
                f"set space_charge_com ds_track_step = {ds_step}",
            ]
        cmds += ["set global track_type = single", "set global track_type = beam"]
        tao.cmds(cmds)
        fout = path / "csr_out.h5"
        tao.cmd(f"write beam -at end {fout}")
        final = ParticleGroup(h5=str(fout))
        input = ImpactXInput.from_tao(tao)
    return final, input


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
def test_csr_cross_code(request: pytest.FixtureRequest, tmp_path: pathlib.Path) -> None:
    """
    Compare output-bunch statistics through a CSR bend: Bmad/Tao vs ImpactX.

    The same bunch is tracked with CSR enabled in both codes; the final beam
    sizes and normalized emittances are compared.  Agreement to a few percent
    validates that ImpactX's CSR reproduces Bmad's 1D CSR on this benchmark.
    """
    kin = 1000.0
    P = _gaussian_bunch(kin, CSR_N_PARTICLES)
    P_final_tao, input = _track_tao_bunch(
        lattice_root / "csr_bench.bmad", P, tmp_path, csr=True
    )
    assert input.csr is True

    for ele in input.lattice:
        if hasattr(ele, "nslice"):
            ele.nslice = max(ele.nslice, 40)
    output = ImpactX(input, workdir=tmp_path).run()

    # ImpactX reduced-beam stats are authoritative (avoid the read-back frame
    # artifact); compare against the Tao final bunch.
    s = output.stats
    comparisons = {
        "sigma_x": (P_final_tao["sigma_x"], s["sigma_x"][-1]),
        "sigma_y": (P_final_tao["sigma_y"], s["sigma_y"][-1]),
        "norm_emit_x": (P_final_tao["norm_emit_x"], s["emittance_xn"][-1]),
        "norm_emit_y": (P_final_tao["norm_emit_y"], s["emittance_yn"][-1]),
    }

    key = max(output.particles, key=lambda k: int(k.split("@")[1]))
    _save_bunch_comparison(
        request.node.name.replace("/", "_"),
        {"Bmad": P_final_tao, "ImpactX": output.particles[key]},
    )

    for name, (tao_val, ix_val) in comparisons.items():
        assert tao_val == pytest.approx(
            ix_val, rel=0.05
        ), f"{name}: Tao={tao_val:.5g} ImpactX={ix_val:.5g}"
