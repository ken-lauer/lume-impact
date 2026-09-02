"""Pydantic models mirroring the ImpactX Python API.

Unlike ImpactZ (which serializes to a text ``ImpactZ.in``), ImpactX has no text
input format: a run is built by instantiating ``impactx.elements.*`` objects and
extending ``sim.lattice``.  These models are a serializable, validated mirror of
that object API; :meth:`InputElement.to_impactx` materializes the live element.
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal, TypeAlias

import pydantic
from impactx import elements

from .constants import ApertureAction, ApertureShape, OpenPMDBackend
from .types import BaseModel, PydanticParticleGroup

# Registry of element models keyed by the upstream ImpactX class name.
input_element_by_class: dict[str, type[InputElement]] = {}


class InputElement(BaseModel):
    """Base class for all ImpactX lattice element models.

    Subclasses declare the ``impactx.elements`` class they mirror via the
    ``impactx_class`` keyword argument, e.g.
    ``class Drift(..., impactx_class=elements.Drift)``.  Subclasses without it
    (e.g. mixins) are not registered.

    Attributes
    ----------
    name : str
        Element name.  Passed through to the ImpactX element and used for
        diagnostics/monitor output naming.
    metadata : dict
        Free-form provenance (e.g. the originating Bmad key and index when
        created via :meth:`ImpactXInput.from_tao`).
    """

    _impactx_class_: ClassVar[type]

    name: str = ""
    metadata: dict[str, int | float | str | bool] = {}

    def __init_subclass__(
        cls, impactx_class: type | None = None, **kwargs: Any
    ) -> None:
        super().__init_subclass__(**kwargs)
        if impactx_class is not None:
            cls._impactx_class_ = impactx_class
            input_element_by_class[impactx_class.__name__] = cls

    def _impactx_kwargs(self) -> dict[str, Any]:
        """Keyword arguments forwarded to the ImpactX element constructor.

        The default gathers every model field except ``metadata`` (and ``name``
        when empty, since ImpactX rejects empty names for some elements).
        """
        kwargs = {
            key: getattr(self, key)
            for key in type(self).model_fields
            if key not in ("metadata", "type")
        }
        if not kwargs.get("name"):
            kwargs.pop("name", None)
        return kwargs

    def to_impactx(self):
        """Instantiate and return the live ``impactx.elements`` object."""
        return self._impactx_class_(**self._impactx_kwargs())


class _Aligned(InputElement):
    """Mixin fields shared by thick, alignable elements."""

    dx: float = 0.0
    dy: float = 0.0
    rotation: float = 0.0
    aperture_x: float = 0.0
    aperture_y: float = 0.0
    nslice: int = 1


class Drift(_Aligned, impactx_class=elements.Drift):
    """A drift.

    Attributes
    ----------
    ds : float
        Segment length [m].
    nslice : int
        Number of slices used for the application of space charge.
    """

    type: Literal["Drift"] = "Drift"
    ds: float = 0.0


class Quad(_Aligned, impactx_class=elements.Quad):
    """A hard-edged, ideal quadrupole.

    Attributes
    ----------
    ds : float
        Segment length [m].
    k : float
        Quadrupole strength [1/m^2] (MAD-X convention); equal to the Bmad ``K1``.
        Positive ``k`` is horizontally focusing.
    """

    type: Literal["Quad"] = "Quad"
    ds: float = 0.0
    k: float = 0.0


class Sbend(_Aligned, impactx_class=elements.Sbend):
    """An ideal sector bend.

    Attributes
    ----------
    ds : float
        Arc length [m].
    rc : float
        Radius of curvature [m] (``= ds / bend_angle``).
    """

    type: Literal["Sbend"] = "Sbend"
    ds: float = 0.0
    rc: float = 0.0


class CFbend(_Aligned, impactx_class=elements.CFbend):
    """An ideal combined-function (dipole + quadrupole) sector bend.

    Attributes
    ----------
    ds : float
        Arc length [m].
    rc : float
        Radius of curvature [m].
    k : float
        Quadrupole strength [1/m^2].
    """

    type: Literal["CFbend"] = "CFbend"
    ds: float = 0.0
    rc: float = 0.0
    k: float = 0.0


class Sol(_Aligned, impactx_class=elements.Sol):
    """An ideal hard-edge solenoid.

    Attributes
    ----------
    ds : float
        Segment length [m].
    ks : float
        Solenoid strength [1/m] (normalized ``B_z``): ``ks = B_z / (B*rho)``.
    """

    type: Literal["Sol"] = "Sol"
    ds: float = 0.0
    ks: float = 0.0


class DipEdge(InputElement, impactx_class=elements.DipEdge):
    """A dipole hard-edge fringe-field map.

    Attributes
    ----------
    psi : float
        Pole face rotation angle [rad].
    rc : float
        Radius of curvature of the associated bend [m].
    g : float
        Full field-region gap [m].
    K2 : float
        Fringe field integral (second field integral).
    location : str
        ``"entry"`` or ``"exit"``.
    """

    type: Literal["DipEdge"] = "DipEdge"
    psi: float = 0.0
    rc: float = 0.0
    g: float = 0.0
    K2: float = 0.0
    location: Literal["entry", "exit"] = "entry"
    dx: float = 0.0
    dy: float = 0.0
    rotation: float = 0.0


class Multipole(InputElement, impactx_class=elements.Multipole):
    """A thin multipole kick of a single order.

    Attributes
    ----------
    multipole : int
        Multipole order (1=dipole, 2=quadrupole, 3=sextupole, ...).
    K_normal : float
        Integrated normal multipole strength.
    K_skew : float
        Integrated skew multipole strength.
    """

    type: Literal["Multipole"] = "Multipole"
    multipole: int = 1
    K_normal: float = 0.0
    K_skew: float = 0.0
    dx: float = 0.0
    dy: float = 0.0
    rotation: float = 0.0


class ExactMultipole(_Aligned, impactx_class=elements.ExactMultipole):
    """A thick multipole using the exact nonlinear Hamiltonian.

    Attributes
    ----------
    ds : float
        Segment length [m].
    k_normal : list of float
        Normal multipole coefficients indexed by order ``n`` (``k_normal[n]``),
        where ``n=0`` is dipole, ``1`` quadrupole, ``2`` sextupole, ....  Units
        are ``1/m^n`` when ``unit=0``.
    k_skew : list of float
        Skew multipole coefficients, same indexing.
    unit : int
        ``0`` for normalized ``1/m^n`` strengths (default), ``1`` for ``T/m^(n-1)``.
    int_order : int
        Symplectic integration order (2, 4, or 6).
    mapsteps : int
        Integration steps per slice.
    """

    type: Literal["ExactMultipole"] = "ExactMultipole"
    ds: float = 0.0
    k_normal: list[float] = pydantic.Field(default_factory=list)
    k_skew: list[float] = pydantic.Field(default_factory=list)
    unit: int = 0
    int_order: int = 2
    mapsteps: int = 5


class RFCavity(_Aligned, impactx_class=elements.RFCavity):
    """An RF cavity with an on-axis field given by a Fourier expansion.

    Attributes
    ----------
    ds : float
        Segment length [m].
    escale : float
        On-axis field scaling [1/m] = (peak on-axis Ez in MV/m) / (rest energy in MeV).
    freq : float
        RF frequency [Hz].
    phase : float
        RF phase [degrees].
    cos_coefficients, sin_coefficients : list of float
        Fourier coefficients of the normalized on-axis longitudinal field Ez(z).
    mapsteps : int
        Integration steps per slice for the reference-orbit ODE.

    Notes
    -----
    The Bmad-to-ImpactX mapping of ``lcavity`` elements currently uses a
    single-harmonic placeholder field profile; trajectory-level agreement with
    Bmad requires a calibrated on-axis field map (future work).
    """

    type: Literal["RFCavity"] = "RFCavity"
    ds: float = 0.0
    escale: float = 0.0
    freq: float = 0.0
    phase: float = 0.0
    cos_coefficients: list[float] = pydantic.Field(default_factory=lambda: [1.0])
    sin_coefficients: list[float] = pydantic.Field(default_factory=lambda: [0.0])
    mapsteps: int = 10


class Kicker(InputElement, impactx_class=elements.Kicker):
    """A thin transverse kicker.

    Attributes
    ----------
    xkick, ykick : float
        Horizontal/vertical kick.  For ``unit="dimensionless"`` these are in
        units of the reference rigidity (i.e. the kick angle ``dpx/p0``); for
        ``unit="T-m"`` they are integrated fields.
    unit : str
        ``"dimensionless"`` (default) or ``"T-m"``.
    """

    type: Literal["Kicker"] = "Kicker"
    xkick: float = 0.0
    ykick: float = 0.0
    unit: Literal["dimensionless", "T-m"] = "dimensionless"
    dx: float = 0.0
    dy: float = 0.0
    rotation: float = 0.0


class Aperture(InputElement, impactx_class=elements.Aperture):
    """A thin collimating aperture."""

    type: Literal["Aperture"] = "Aperture"
    aperture_x: float = 0.0
    aperture_y: float = 0.0
    shape: ApertureShape = ApertureShape.rectangular
    action: ApertureAction = ApertureAction.transmit
    dx: float = 0.0
    dy: float = 0.0
    rotation: float = 0.0

    def _impactx_kwargs(self) -> dict[str, Any]:
        kwargs = super()._impactx_kwargs()
        kwargs["shape"] = self.shape.value
        kwargs["action"] = self.action.value
        return kwargs


class Marker(InputElement, impactx_class=elements.Marker):
    """A zero-length marker with no effect on the beam."""

    type: Literal["Marker"] = "Marker"

    def _impactx_kwargs(self) -> dict[str, Any]:
        return {"name": self.name or "marker"}


class BeamMonitor(InputElement, impactx_class=elements.BeamMonitor):
    """A zero-length diagnostic that writes the beam to openPMD.

    Attributes
    ----------
    backend : OpenPMDBackend
        openPMD file backend (``h5`` recommended for lume-impact round-trips).
    """

    type: Literal["BeamMonitor"] = "BeamMonitor"
    backend: OpenPMDBackend = OpenPMDBackend.h5
    encoding: Literal["g", "f", "v"] = "g"
    period_sample_intervals: int = 1

    def _impactx_kwargs(self) -> dict[str, Any]:
        return {
            "name": self.name or "monitor",
            "backend": self.backend.value,
            "encoding": self.encoding,
            "period_sample_intervals": self.period_sample_intervals,
        }


AnyInputElement: TypeAlias = (
    Drift
    | Quad
    | Sbend
    | CFbend
    | Sol
    | DipEdge
    | Multipole
    | ExactMultipole
    | RFCavity
    | Kicker
    | Aperture
    | Marker
    | BeamMonitor
)


class ImpactXInput(BaseModel):
    """A full ImpactX simulation description.

    Attributes
    ----------
    lattice : list of InputElement
        Ordered beamline, including any :class:`BeamMonitor` diagnostics.
    species : str
        Reference particle species (e.g. ``"electron"``, ``"positron"``, ``"proton"``).
    kin_energy_MeV : float
        Reference particle kinetic energy [MeV].
    mass_MeV : float or None
        Reference particle rest energy [MeV].  If None, ImpactX uses the built-in
        value for ``species``.
    charge_qe : float or None
        Reference particle charge in units of the elementary charge.
    initial_particles : ParticleGroup or None
        Explicit initial distribution.  When set it is loaded directly into the
        ImpactX particle container (exact distribution) rather than sampled from
        a statistical distribution.
    bunch_charge_C : float
        Total bunch charge [C]; used for space charge and particle weighting.
    n_particle : int
        Number of macroparticles (only used when sampling a distribution).
    space_charge : bool
        Enable the space-charge solver.
    csr : bool
        Enable coherent synchrotron radiation (a collective effect applied in
        bends).  Requires a multi-particle bunch and a transverse mesh.
    csr_bins : int
        Number of longitudinal bins for the CSR calculation.
    particle_shape : int
        B-spline deposition order (1, 2, or 3) used for collective effects.
    slice_step_diagnostics : bool
        Emit per-slice reduced-beam diagnostics (the per-``s`` stats table).
    """

    lattice: list[AnyInputElement] = pydantic.Field(default_factory=list)

    species: str = "electron"
    kin_energy_MeV: float = 0.0
    mass_MeV: float | None = None
    charge_qe: float | None = None

    initial_particles: PydanticParticleGroup | None = None
    bunch_charge_C: float = 0.0
    n_particle: int = 10000

    space_charge: bool = False
    csr: bool = False
    csr_bins: int = 150
    particle_shape: int = 2
    slice_step_diagnostics: bool = True

    def to_impactx_lattice(self) -> list:
        """Return the list of live ``impactx.elements`` objects."""
        return [ele.to_impactx() for ele in self.lattice]

    @classmethod
    def from_tao(cls, tao, **kwargs: Any) -> ImpactXInput:
        """Create an :class:`ImpactXInput` from a live Tao instance's lattice.

        Parameters
        ----------
        tao : pytao.Tao
            A running Tao instance; live element parameters are read from it.
        track_start, track_end : str, optional
            Element names bounding the range to convert.
        which : {"model", "base", "design"}
            Which lattice values to read (default "model").
        add_monitors : bool
            Insert a :class:`BeamMonitor` after each thick element (default True).
        """
        from .interfaces.bmad import impactx_input_from_tao

        return impactx_input_from_tao(tao, **kwargs)
