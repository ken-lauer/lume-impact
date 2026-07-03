"""Conversion between :class:`beamphysics.ParticleGroup` and ImpactX.

ImpactX tracks particles in fixed-``s`` coordinates ``(x, y, t, px, py, pt)``
relative to a reference particle, with transverse momenta normalized by the
reference momentum ``p0 = m c beta0 gamma0``.  A beamphysics ``ParticleGroup``
stores lab-frame positions [m] and momenta [eV/c] at a common time.

The coordinate transforms below mirror those distributed with ImpactX in
``examples/initialize_from_array/transformation_utilities.py``.
"""

from __future__ import annotations

import numpy as np
from beamphysics import ParticleGroup
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
