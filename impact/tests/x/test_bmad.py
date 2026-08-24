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


def _chirp_phase(P: ParticleGroup) -> float:
    """
    Longitudinal chirp as an angle [rad] from the ``z``-``delta`` correlation.
    """
    z = np.asarray(P["z"], dtype=float) - P["mean_z"]
    delta = (np.asarray(P["p"], dtype=float) - P["mean_p"]) / P["mean_p"]
    zz = float(np.mean(z * z))
    return float(np.arctan2(np.mean(z * delta), zz)) if zz > 0 else 0.0


def pplot(
    P: ParticleGroup,
    xkey: str,
    ykey: str,
    text: str | None = None,
    title: str | None = None,
    **kwargs,
):
    """
    Plot ParticleGroup phase space with standardized beam-statistics annotations.

    A thin wrapper around :meth:`ParticleGroup.plot` that annotates common
    projections (longitudinal phase space, transverse spot, x-xp, y-yp) with the
    relevant beam statistics.  Returns the figure when ``return_figure=True``.
    """
    from matplotlib.gridspec import GridSpec

    return_figure = kwargs.get("return_figure", False)
    kwargs["return_figure"] = True
    fig = P.plot(xkey, ykey, **kwargs)

    if text is None:
        auto_text = None
        if xkey in ("delta_z/c", "z/c") and ykey == "energy":
            sigma_z = P["sigma_z"]
            sigma_p = P["sigma_p"]
            p0 = P["mean_p"]
            auto_text = (
                rf"$\sigma_z/c$= {sigma_z / c_light * 1e15:.0f} fs"
                "\n"
                rf"$\sigma_\delta$= {sigma_p / p0 * 1e4:.1f}$\times10^{{-4}}$"
                "\n"
                rf"chirp= ${_chirp_phase(P) * 180 / np.pi:.1f}^\circ$"
                "\n"
                rf"$\left<E\right>$= {P['mean_energy'] / 1e6:.1f} MeV"
            )
        elif xkey == "x" and ykey == "y":
            auto_text = (
                rf"$\sigma_x$= {P['sigma_x'] * 1e6:.1f} µm"
                "\n"
                rf"$\sigma_y$= {P['sigma_y'] * 1e6:.1f} µm"
            )
        elif xkey == "x" and ykey in ("xp", "px"):
            auto_text = r"$\epsilon_{n,x}$" + f"\n{P['norm_emit_x'] * 1e6:.2f} mm-mrad"
        elif xkey == "y" and ykey in ("yp", "py"):
            auto_text = r"$\epsilon_{n,y}$" + f"\n{P['norm_emit_y'] * 1e6:.2f} mm-mrad"
        text = auto_text

    if text:
        gs = GridSpec(4, 4, figure=fig)
        ax_text = fig.add_subplot(gs[0, -1])
        ax_text.axis("off")
        ax_text.text(0.5, 0.5, text, fontsize=10, ha="center", va="center")
    if title and fig and len(fig.axes) > 1:
        fig.axes[1].set_title(title)
    return fig if return_figure else None


# Standard phase-space views (xkey, ykey, title, extra kwargs).
PHASE_SPACE_VIEWS = (
    ("delta_z/c", "energy", "Longitudinal phase space", {}),
    ("x", "y", "Transverse spot", {}),
    ("x", "xp", "Horizontal phase space", {"ellipse": True}),
    ("y", "yp", "Vertical phase space", {"ellipse": True}),
)


def _save_phase_space_views(name: str, P: ParticleGroup) -> None:
    """Save the standard pplot phase-space views of a bunch to ``artifacts/``.

    The bunch is drifted to a common time so its longitudinal extent lives in
    ``z`` (fixed-s bunches store it in ``t``, leaving ``delta_z/c`` degenerate).
    """
    P = P.copy()
    P.drift_to_t()
    test_artifacts.mkdir(parents=True, exist_ok=True)
    for xkey, ykey, title, kw in PHASE_SPACE_VIEWS:
        try:
            fig = pplot(P, xkey, ykey, title=title, return_figure=True, **kw)
        except Exception:
            continue
        tag = f"{xkey}_{ykey}".replace("/", "")
        fig.savefig(test_artifacts / f"{name}_{tag}.png", dpi=100, bbox_inches="tight")
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
                "set bmad_com csr_and_space_charge_on = T",
                "set ele sbend::* csr_method = 1_dim",
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


# CSR is physical only in bends (straight elements emit none), and only a
# suitable energy/charge regime is a meaningful cross-code benchmark: a generic
# low-energy dipole (e.g. dipole.bmad at 10 MeV) drives CSR nonperturbative,
# where Bmad's 1D CSR and ImpactX's models legitimately diverge.  These are the
# designed high-energy CSR benchmarks (single bend; 5 GeV chicane).
csr_bend_lattices = ["csr_bench.bmad", "csr_zeuthen.bmad"]


@pytest.mark.slow
@pytest.mark.parametrize("lattice", csr_bend_lattices)
def test_csr_cross_code(
    request: pytest.FixtureRequest, tmp_path: pathlib.Path, lattice: str
) -> None:
    """
    Compare output-bunch statistics through a CSR bend: Bmad/Tao vs ImpactX.

    The same bunch is tracked with CSR enabled in both codes; the final beam
    sizes and normalized emittances are compared, and phase-space figures for
    both bunches are written to ``artifacts/``.  Agreement to a few percent
    validates that ImpactX's CSR reproduces Bmad's 1D CSR.

    Statistics use ImpactX's authoritative reduced-beam diagnostics; the monitor
    read-back (used only for the plots) scales momentum-spread quantities by
    cos(bend angle), a known BeamMonitor-output artifact.
    """
    latt = lattice_root / lattice
    with Tao(lattice_file=str(latt), noplot=True) as tao:
        kin = ImpactXInput.from_tao(tao).kin_energy_MeV
    P = _gaussian_bunch(kin, CSR_N_PARTICLES)

    P_final_tao, input = _track_tao_bunch(latt, P, tmp_path, csr=True)
    assert input.csr is True

    for ele in input.lattice:
        if hasattr(ele, "nslice"):
            ele.nslice = max(ele.nslice, 40)
    output = ImpactX(input, workdir=tmp_path).run()

    s = output.stats
    comparisons = {
        "sigma_x": (P_final_tao["sigma_x"], s["sigma_x"][-1]),
        "sigma_y": (P_final_tao["sigma_y"], s["sigma_y"][-1]),
        "norm_emit_x": (P_final_tao["norm_emit_x"], s["emittance_xn"][-1]),
        "norm_emit_y": (P_final_tao["norm_emit_y"], s["emittance_yn"][-1]),
    }

    name = request.node.name.replace("/", "_")
    key = max(output.particles, key=lambda k: int(k.split("@")[1]))
    P_final_ix = output.particles[key]
    _save_bunch_comparison(name, {"Bmad": P_final_tao, "ImpactX": P_final_ix})
    _save_phase_space_views(f"{name}_bmad", P_final_tao)
    _save_phase_space_views(f"{name}_impactx", P_final_ix)

    for stat, (tao_val, ix_val) in comparisons.items():
        assert tao_val == pytest.approx(
            ix_val, rel=0.05
        ), f"{stat}: Tao={tao_val:.5g} ImpactX={ix_val:.5g}"
