"""Conversion of a live Tao lattice into an :class:`ImpactXInput`.

Mirrors ``impact.z.interfaces.bmad`` but targets the ImpactX element API.  The
converter reads live element parameters from a running :class:`pytao.Tao`
instance (via ``ele_info``) and dispatches on the Bmad element key.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Literal

from beamphysics import ParticleGroup
from beamphysics.particles import c_light
from beamphysics.species import charge_state
from pytao import Tao, TaoCommandError

from ...interfaces.bmad import ele_info

# Reuse the generic (tool-agnostic) Tao helpers from the ImpactZ interface.
from ...z.interfaces.bmad import export_particles, get_index_to_name
from ..input import (
    AnyInputElement,
    BeamMonitor,
    CFbend,
    DipEdge,
    Drift,
    ExactMultipole,
    ImpactXInput,
    Kicker,
    Marker,
    Quad,
    RFCavity,
    Sbend,
    Sol,
)

logger = logging.getLogger(__name__)

Which = Literal["model", "base", "design"]

DRIFT_ELEMENT_KEYS = {
    "drift",
    "pipe",
    "instrument",
    "ecollimator",
    "rcollimator",
    "monitor",
}
# Bmad keys that carry no field and are represented as zero-length markers.
MARKER_ELEMENT_KEYS = {"marker", "fork", "photon_fork", "beginning_ele"}

# Native multipole elements -> (multipole index n, Bmad field-gradient key).
# The index is the ExactMultipole ``k_normal`` list position: n=2 sextupole,
# n=3 octupole (n=0 dipole, n=1 quadrupole).
MULTIPOLE_GRADIENT = {
    "sextupole": (2, "B2_GRADIENT"),
    "octupole": (3, "B3_GRADIENT"),
}

# Number of hard-edge slices per wiggler/undulator period.  Each slice is a
# thin horizontal kick about a straight reference; the trajectory converges to
# Bmad's with increasing resolution (~0.4% of orbit amplitude at 20).
WIGGLER_SLICES_PER_PERIOD = 20


def _normalized_strength(charge: float, p0c: float, gradient: float) -> float:
    """Normalized multipole strength ``K_n`` [1/m^n] from a field gradient.

    ``K_n = q * (c / P0C) * (d^n B / dx^n)`` (charge sign included).
    """
    return charge * (c_light / p0c) * gradient


class UnsupportedElementError(NotImplementedError):
    """Raised when a Bmad element has no ImpactX representation yet."""


def _f(info: dict, key: str, default: float = 0.0) -> float:
    return float(info.get(key, default))


def single_element_from_tao_info(
    info: dict[str, Any],
    *,
    name: str = "",
    species: str = "electron",
    multipoles: list[dict[str, Any]] | None = None,
) -> list[AnyInputElement]:
    """Convert one Tao element's info dict into ImpactX element model(s).

    Returns a list because some Bmad elements expand into several ImpactX
    elements (e.g. an ``sbend`` with pole-face rotations becomes
    ``DipEdge, Sbend, DipEdge``, or a thick kicker becomes ``Drift, Kicker,
    Drift``).

    Parameters
    ----------
    info : dict
        Element attributes from :func:`ele_info`.
    name : str
        Element name.
    species : str
        Reference species, used to sign normalized multipole strengths.
    multipoles : list of dict, optional
        ``ele_multipoles`` data for a ``thick_multipole`` element.
    """
    key = str(info["key"]).lower()
    length = _f(info, "L")
    num_steps = int(info.get("NUM_STEPS", 1) or 1)
    metadata = {"bmad_key": key}
    charge = charge_state(species)

    common = dict(
        name=name,
        nslice=num_steps,
        dx=_f(info, "X_OFFSET"),
        dy=_f(info, "Y_OFFSET"),
        rotation=-_f(info, "TILT"),
        metadata=metadata,
    )

    if key in MARKER_ELEMENT_KEYS and length == 0.0:
        return [Marker(name=name, metadata=metadata)]

    if key in DRIFT_ELEMENT_KEYS:
        return [Drift(ds=length, **common)]

    if key == "quadrupole":
        return [Quad(ds=length, k=_f(info, "K1"), **common)]

    if key in {"hkicker", "vkicker", "kicker"}:
        return _kicker_from_info(info, key, length, common, metadata)

    if key == "solenoid":
        # ImpactX ks == Bmad normalized KS [1/m] (= q * B_z / P0C * c).
        ks = _f(info, "KS")
        if ks == 0.0:
            p0c = _f(info, "P0C")
            brho = p0c / c_light if p0c else 0.0
            ks = charge * _f(info, "BS_FIELD") / brho if brho else 0.0
        return [Sol(ds=length, ks=ks, **common)]

    if key == "sbend":
        angle = _f(info, "ANGLE")
        if angle == 0.0 or length == 0.0:
            return [Drift(ds=length, **common)]
        rc = length / angle
        k1 = _f(info, "K1")
        e1 = _f(info, "E1")
        e2 = _f(info, "E2")
        hgap = _f(info, "HGAP")
        fint = _f(info, "FINT")
        g = 2.0 * hgap
        body_cls = CFbend if k1 != 0.0 else Sbend
        body_kwargs = dict(ds=length, rc=rc, **common)
        if k1 != 0.0:
            body_kwargs["k"] = k1
        body = body_cls(**body_kwargs)

        elements: list[AnyInputElement] = [body]
        if e1 != 0.0:
            elements.insert(
                0,
                DipEdge(
                    name=name,
                    psi=e1,
                    rc=rc,
                    g=g,
                    K2=fint,
                    location="entry",
                    metadata=metadata,
                ),
            )
        if e2 != 0.0:
            elements.append(
                DipEdge(
                    name=name,
                    psi=e2,
                    rc=rc,
                    g=g,
                    K2=fint,
                    location="exit",
                    metadata=metadata,
                )
            )
        return elements

    if key in MULTIPOLE_GRADIENT:
        # Native sextupole/octupole -> thick ExactMultipole with a single order.
        index, grad_key = MULTIPOLE_GRADIENT[key]
        kn = _normalized_strength(charge, _f(info, "P0C"), _f(info, grad_key))
        k_normal = [0.0] * (index + 1)
        k_normal[index] = kn
        return [
            ExactMultipole(
                ds=length,
                k_normal=k_normal,
                k_skew=[0.0] * (index + 1),
                **common,
            )
        ]

    if key == "thick_multipole":
        return [_thick_multipole_from_info(length, common, multipoles)]

    if key == "lcavity":
        return [_rfcavity_from_info(info, length, common, species)]

    if key == "wiggler":
        return _wiggler_from_info(info, length, name, metadata)

    if length > 0.0:
        raise UnsupportedElementError(f"{key!r} (length {length} m) is not supported")
    # Unknown zero-length element -> harmless marker.
    return [Marker(name=name, metadata=metadata)]


def _kicker_from_info(
    info: dict[str, Any],
    key: str,
    length: float,
    common: dict[str, Any],
    metadata: dict[str, Any],
) -> list[AnyInputElement]:
    """Map a Bmad (h/v)kicker to a thin ImpactX Kicker, centered in a drift.

    Bmad kicks (``KICK``/``HKICK``/``VKICK``) are dimensionless kick angles
    ``dp/p0``, matching ImpactX's ``unit="dimensionless"``.
    """
    if key == "hkicker":
        xkick, ykick = _f(info, "KICK"), 0.0
    elif key == "vkicker":
        xkick, ykick = 0.0, _f(info, "KICK")
    else:
        xkick, ykick = _f(info, "HKICK"), _f(info, "VKICK")

    name = common["name"]
    kicker = Kicker(
        name=name,
        xkick=xkick,
        ykick=ykick,
        dx=common["dx"],
        dy=common["dy"],
        rotation=common["rotation"],
        metadata=metadata,
    )
    if length == 0.0:
        return [kicker]
    half = Drift(ds=length / 2.0, name=name, metadata=metadata)
    return [half, kicker, half.model_copy()]


def _thick_multipole_from_info(
    length: float,
    common: dict[str, Any],
    multipoles: list[dict[str, Any]] | None,
) -> ExactMultipole:
    """Map a Bmad ``thick_multipole`` to an ImpactX ExactMultipole.

    Uses Bmad's per-order ``KnL (equiv)`` integrated normalized strengths,
    converting to the per-metre coefficients ImpactX expects.
    """
    multipoles = multipoles or []
    max_index = max((int(m["index"]) for m in multipoles), default=0)
    k_normal = [0.0] * (max_index + 1)
    k_skew = [0.0] * (max_index + 1)
    for m in multipoles:
        n = int(m["index"])
        knl = float(m.get("KnL (equiv)", 0.0))
        bn = float(m.get("Bn", 0.0))
        an = float(m.get("An", 0.0))
        k_normal[n] = knl / length if length else 0.0
        # Skew shares the normal's normalization; An/Bn carries the skew fraction.
        if bn:
            k_skew[n] = (knl * an / bn) / length if length else 0.0
    return ExactMultipole(ds=length, k_normal=k_normal, k_skew=k_skew, **common)


def _rfcavity_from_info(
    info: dict[str, Any],
    length: float,
    common: dict[str, Any],
    species: str,
) -> RFCavity:
    """Map a Bmad ``lcavity`` to an ImpactX RFCavity.

    Approximate: ``escale`` is derived from the Bmad ``GRADIENT`` and a
    single-harmonic on-axis profile is assumed.  Trajectory-level agreement
    with Bmad requires a calibrated field map and phase-convention handling
    (future work); ``from_tao`` produces a valid element so lattices with RF
    cavities convert.
    """
    from beamphysics.species import mass_of

    rest_energy_MeV = mass_of(species) / 1e6
    gradient_MV_per_m = _f(info, "GRADIENT") / 1e6
    escale = abs(gradient_MV_per_m) / rest_energy_MeV
    return RFCavity(
        name=common["name"],
        ds=length,
        escale=escale,
        freq=_f(info, "RF_FREQUENCY"),
        phase=_f(info, "PHI0") * 360.0,
        nslice=common["nslice"],
        dx=common["dx"],
        dy=common["dy"],
        rotation=common["rotation"],
        metadata=common["metadata"],
    )


def _wiggler_from_info(
    info: dict[str, Any],
    length: float,
    name: str,
    metadata: dict[str, Any],
) -> list[AnyInputElement]:
    """Model a planar wiggler/undulator as a series of hard-edge kicks.

    ImpactX has no wiggler element, so the periodic vertical field
    ``B_y(s) = B_max * cos(k_u s)`` is represented as thin horizontal
    :class:`Kicker` slices about a straight reference (each with the slice's
    integrated bend angle ``g(s) ds``), bracketed by half drifts.  This keeps the
    design orbit straight -- matching Bmad -- while reproducing the oscillating
    trajectory; it converges to Bmad's orbit as the slice count increases.
    """
    l_period = _f(info, "L_PERIOD")
    n_period = int(_f(info, "N_PERIOD"))
    g_max = _f(info, "G_MAX")  # max curvature 1/m = B_max / (P0C/c), reference-signed
    polarity = _f(info, "POLARITY", 1.0)
    if l_period <= 0.0 or n_period <= 0 or g_max == 0.0:
        return [Drift(ds=length, name=name, metadata=metadata)]

    ku = 2.0 * math.pi / l_period
    n_slices = n_period * WIGGLER_SLICES_PER_PERIOD
    ds = length / n_slices
    elements: list[AnyInputElement] = []
    for i in range(n_slices):
        s_center = (i + 0.5) * ds
        xkick = polarity * g_max * math.cos(ku * s_center) * ds
        elements += [
            Drift(ds=ds / 2.0, name=name, metadata=metadata),
            Kicker(xkick=xkick, ykick=0.0, name=name, metadata=metadata),
            Drift(ds=ds / 2.0, name=name, metadata=metadata),
        ]
    return elements


def element_from_tao(
    tao: Tao,
    ele_id: str | int,
    which: Which = "model",
    name: str = "",
    species: str = "electron",
) -> list[AnyInputElement]:
    """Read one Tao element and convert it to ImpactX element model(s).

    Elements Tao cannot describe as beamline elements (e.g. the ``beginning``
    pseudo-element, which lacks a length) are silently skipped.
    """
    try:
        info = ele_info(tao, ele_id, which=which)
    except KeyError:
        return []

    multipoles = None
    if str(info["key"]).lower() == "thick_multipole":
        mp = tao.ele_multipoles(ele_id)
        if mp.get("multipoles_on"):
            multipoles = mp.get("data") or None

    return single_element_from_tao_info(
        info, name=name, species=species, multipoles=multipoles
    )


@dataclass
class ImpactXConversionState:
    """Intermediate state extracted from a Tao lattice for conversion."""

    idx_to_name: dict[int, str]
    species: str
    kin_energy_MeV: float
    initial_particles: ParticleGroup | None
    csr: bool = False
    which: Which = "model"

    @classmethod
    def from_tao(
        cls,
        tao: Tao,
        track_start: str | None = None,
        track_end: str | None = None,
        ix_uni: int = 1,
        ix_branch: int = 0,
        which: Which = "model",
    ) -> ImpactXConversionState:
        idx_to_name = get_index_to_name(
            tao,
            track_start=track_start,
            track_end=track_end,
            ix_uni=ix_uni,
            ix_branch=ix_branch,
        )
        ix_beginning = list(idx_to_name)[0]

        branch1 = dict(tao.branch1(ix_uni, ix_branch))
        species = str(branch1["param_particle"])

        from beamphysics.species import mass_of

        start_attrs = dict(tao.ele_gen_attribs(str(ix_beginning), which=which))
        e_tot = float(start_attrs["E_TOT"])  # total energy [eV]
        mass_eV = mass_of(species.lower())
        kin_energy_MeV = (e_tot - mass_eV) / 1e6

        try:
            initial_particles = export_particles(tao, ix_beginning)
        except TaoCommandError as ex:
            logger.warning("No initial particles exported: %s", ex)
            initial_particles = None

        csr = bool(dict(tao.bmad_com()).get("csr_and_space_charge_on", False))

        return cls(
            idx_to_name=idx_to_name,
            species=species.lower(),
            kin_energy_MeV=kin_energy_MeV,
            initial_particles=initial_particles,
            csr=csr,
            which=which,
        )

    def convert_lattice(
        self,
        tao: Tao,
        add_monitors: bool = True,
    ) -> list[AnyInputElement]:
        """Convert every in-range element into the ImpactX lattice."""
        lattice: list[AnyInputElement] = []
        if add_monitors:
            lattice.append(BeamMonitor(name="initial"))
        for ele_id, name in self.idx_to_name.items():
            elements = element_from_tao(
                tao, ele_id, which=self.which, name=name, species=self.species
            )
            lattice.extend(elements)
            if add_monitors and any(getattr(e, "ds", 0.0) for e in elements):
                lattice.append(BeamMonitor(name=name or f"ele_{ele_id}"))
        return lattice

    def to_input(self, lattice: list[AnyInputElement]) -> ImpactXInput:
        return ImpactXInput(
            lattice=lattice,
            species=self.species,
            kin_energy_MeV=self.kin_energy_MeV,
            initial_particles=self.initial_particles,
            csr=self.csr,
            bunch_charge_C=(
                self.initial_particles.charge
                if self.initial_particles is not None
                else 0.0
            ),
        )


def impactx_input_from_tao(
    tao: Tao,
    track_start: str | None = None,
    track_end: str | None = None,
    *,
    ix_uni: int = 1,
    ix_branch: int = 0,
    which: Which = "model",
    add_monitors: bool = True,
) -> ImpactXInput:
    """Create an :class:`ImpactXInput` from a live Tao lattice."""
    state = ImpactXConversionState.from_tao(
        tao,
        track_start=track_start,
        track_end=track_end,
        ix_uni=ix_uni,
        ix_branch=ix_branch,
        which=which,
    )
    lattice = state.convert_lattice(tao, add_monitors=add_monitors)
    return state.to_input(lattice)
