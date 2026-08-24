"""Conversion between :class:`beamphysics.ParticleGroup` and ImpactX.

ImpactX tracks particles in fixed-``s`` coordinates ``(x, y, t, px, py, pt)``
relative to a reference particle, with transverse momenta normalized by the
reference momentum ``p0 = m c beta0 gamma0``.  A beamphysics ``ParticleGroup``
stores lab-frame positions [m] and momenta [eV/c] at a common time.

beamphysics ships interfaces for many codes (astra, elegant, gpt, impact,
bmad, ...) but *not* ImpactX, so the coordinate maps here are ports of
ImpactX's own ``examples/initialize_from_array/transformation_utilities.py`` --
the authoritative ImpactX convention -- rather than reimplementations of
beamphysics functionality.  Note in particular that ImpactX's longitudinal
coordinate ``pt = -d(gamma)/(beta0 gamma0)`` differs from Bmad's
``pz = d(beta*gamma)/(beta0 gamma0)``, so ``ParticleGroup.to_bmad`` cannot
supply it; only the transverse coordinates coincide with Bmad's.
"""

from __future__ import annotations

import numpy as np
from beamphysics import ParticleGroup
from beamphysics.particles import c_light
from beamphysics.species import charge_state

# elementary charge [C]
Q_E = 1.602176634e-19


def _to_podvector(arr):
    """Convert a 1-D array to an AMReX ``PODVector_real_std``.

    Newer ImpactX accepts numpy arrays for ``add_n_particles`` directly; older
    releases require an explicit PODVector.  This handles both.
    """
    import amrex.space3d as amr

    arr = np.ascontiguousarray(np.asarray(arr, dtype=float))
    try:
        return amr.PODVector_real_std(arr)
    except (TypeError, RuntimeError):
        vec = amr.PODVector_real_std(len(arr))
        np.asarray(vec)[:] = arr
        return vec


def _to_ref_part_t(ref, x, y, z, px, py, pz):
    """Lab-frame arrays -> deviations from the reference particle (fixed t)."""
    dx = x - ref.x
    dy = y - ref.y
    dz = z - ref.z
    dpx = (px - ref.px) / ref.pz
    dpy = (py - ref.py) / ref.pz
    dpz = (pz - ref.pz) / ref.pz
    return dx, dy, dz, dpx, dpy, dpz


def _to_s_from_t(ref, dx, dy, dz, dpx, dpy, dpz):
    """Fixed-t deviations -> fixed-s deviations ``(x, y, t, px, py, pt)``."""
    ref_pz = ref.pz
    ref_pt = ref.pt
    dxs = dx - ref_pz * dpx * dz / (ref_pz + ref_pz * dpz)
    dys = dy - ref_pz * dpy * dz / (ref_pz + ref_pz * dpz)
    pt = -np.sqrt(
        1.0 + (ref_pz + ref_pz * dpz) ** 2 + (ref_pz * dpx) ** 2 + (ref_pz * dpy) ** 2
    )
    dt = pt * dz / (ref_pz + ref_pz * dpz)
    dpt = (pt - ref_pt) / ref_pz
    return dxs, dys, dt, dpx, dpy, dpt


def _species_from_mass_charge(mass_kg: float, charge_C: float) -> str:
    """Best-effort particle species from reference mass [kg] and charge [C]."""
    m_p = 1.67262192369e-27
    if abs(mass_kg - m_p) / m_p < 0.05:
        return "proton"
    # electron-mass particle: sign of charge distinguishes electron/positron
    return "positron" if charge_C > 0 else "electron"


def impactx_monitor_to_particle_group(group) -> ParticleGroup:
    """Convert an ImpactX openPMD ``beam`` group to a lab-frame ParticleGroup.

    ImpactX stores fixed-``s`` phase space relative to the reference particle:
    ``x, y`` [m], ``t = c*dt`` [m], and momenta ``px, py = p_{x,y}/p0`` and
    ``pt = -d(gamma)/(beta0 gamma0)`` normalized by the reference momentum.  Since
    ImpactX's transverse coordinates coincide with Bmad's, the phase space maps
    directly onto Bmad coordinates and :meth:`ParticleGroup.from_bmad` performs
    the lab-frame reconstruction (avoiding a hand-rolled frame transform).

    Parameters
    ----------
    group : h5py.Group
        The ``.../particles/beam`` group written by an ImpactX ``BeamMonitor``.
    """
    attrs = group.attrs
    beta0_gamma0 = float(attrs["pz_ref"])  # reference beta*gamma
    gamma0 = np.hypot(1.0, beta0_gamma0)
    mass_kg = float(attrs["mass_ref"])
    charge_ref_C = float(attrs["charge_ref"])
    mc2_eV = mass_kg * c_light**2 / Q_E
    p0c = beta0_gamma0 * mc2_eV  # reference momentum * c [eV]
    species = _species_from_mass_charge(mass_kg, charge_ref_C)

    pos, mom = group["position"], group["momentum"]
    x = np.asarray(pos["x"], dtype=float)
    y = np.asarray(pos["y"], dtype=float)
    ct = np.asarray(pos["t"], dtype=float)  # c * (t - t_ref)
    px = np.asarray(mom["x"], dtype=float)  # = px / p0  == Bmad px
    py = np.asarray(mom["y"], dtype=float)
    pt = np.asarray(mom["t"], dtype=float)  # = -d(gamma)/(beta0 gamma0)

    # ImpactX pt -> gamma -> Bmad pz = p/p0 - 1
    gamma = gamma0 - pt * beta0_gamma0
    beta_gamma = np.sqrt(np.clip(gamma**2 - 1.0, 0.0, None))
    pz_bmad = beta_gamma / beta0_gamma0 - 1.0
    beta = beta_gamma / gamma
    # Bmad z = -beta * c * dt = -beta * (c*dt)
    z_bmad = -beta * ct

    weighting = np.asarray(group["weighting"], dtype=float)
    bmad = {
        "x": x,
        "px": px,
        "y": y,
        "py": py,
        "z": z_bmad,
        "pz": pz_bmad,
        "p0c": p0c,
        "tref": 0.0,
        "species": species,
        "charge": weighting * abs(charge_ref_C),
        "state": np.ones(len(x), dtype=int),
    }
    return ParticleGroup.from_bmad(bmad)


def particle_group_to_impactx(
    beam, ref, P: ParticleGroup, bunch_charge_C: float | None = None
) -> None:
    """Load a ``ParticleGroup`` into an ImpactX beam container.

    Parameters
    ----------
    beam : impactx ParticleContainer
        The target container (``sim.particle_container()``).
    ref : impactx RefPart
        Reference particle, already configured with species/energy.  Its ``z``
        is used as the longitudinal origin.
    P : ParticleGroup
        The distribution to load.  Momenta are converted from [eV/c] to the
        reference-normalized ``beta gamma`` used by ImpactX.
    bunch_charge_C : float, optional
        Total bunch charge [C].  If None, ``P.charge`` is used.  Per-particle
        weights follow ``P.weight``.
    """
    mc2 = P.mass  # rest energy [eV]; P.px/mc2 == (beta gamma)_x

    # Normalize to a common-time snapshot: the fixed-s -> fixed-s ImpactX transform
    # below (`_to_s_from_t`) expects a fixed-t bunch (spread in z), whereas an
    # accelerator bunch (e.g. exported from Tao) is at fixed s (spread in t,
    # z ~ 0).  drift_to_t drifts every particle to the mean time so the
    # longitudinal extent lives in z, as the transform requires.  It is a no-op
    # for a bunch already at common time.  (A tolerance test on t is unreliable:
    # absolute times ~1e-13 s look "equal" to np.allclose while carrying the full
    # bunch length.)
    if len(P) > 1:
        P = P.copy()
        P.drift_to_t()

    x = np.asarray(P.x, dtype=float)
    y = np.asarray(P.y, dtype=float)
    z = np.asarray(P.z, dtype=float)
    px = np.asarray(P.px, dtype=float) / mc2
    py = np.asarray(P.py, dtype=float) / mc2
    pz = np.asarray(P.pz, dtype=float) / mc2

    dx, dy, dz, dpx, dpy, dpz = _to_ref_part_t(ref, x, y, z, px, py, pz)
    dx, dy, dt, dpx, dpy, dpt = _to_s_from_t(ref, dx, dy, dz, dpx, dpy, dpz)

    # charge/mass ratio in e / eV: (charge in units of e) / (rest energy in eV).
    qm_eev = charge_state(P.species) / mc2

    # ImpactX ``w`` is the number of physical particles per macroparticle.
    # ParticleGroup weight is absolute charge [C], so divide by |e|.
    w = np.asarray(P.weight, dtype=float) / Q_E
    if bunch_charge_C is not None:
        w *= (bunch_charge_C / Q_E) / w.sum()

    beam.add_n_particles(
        _to_podvector(dx),
        _to_podvector(dy),
        _to_podvector(dt),
        _to_podvector(dpx),
        _to_podvector(dpy),
        _to_podvector(dpt),
        qm_eev,
        w=_to_podvector(w),
    )
