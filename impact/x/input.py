"""Pydantic models mirroring the ImpactX Python API.

Unlike ImpactZ (which serializes to a text ``ImpactZ.in``), ImpactX has no text
input format: a run is built by instantiating ``impactx.elements.*`` objects and
extending ``sim.lattice``.  These models are a serializable, validated mirror of
that object API; :meth:`InputElement.to_impactx` materializes the live element.
"""

from __future__ import annotations

from typing import Any, ClassVar, Union

import pydantic
from typing_extensions import Literal, TypeAlias

from .constants import ApertureAction, ApertureShape, OpenPMDBackend
from .types import BaseModel, PydanticParticleGroup

# Registry of element models keyed by the upstream ImpactX class name.
input_element_by_class: dict[str, type["InputElement"]] = {}


class InputElement(BaseModel):
    """Base class for all ImpactX lattice element models.

    Subclasses declare the name of the ImpactX element class they mirror via the
    ``impactx_class`` keyword argument, e.g. ``class Drift(..., impactx_class="Drift")``.

    Attributes
    ----------
    name : str
        Element name.  Passed through to the ImpactX element and used for
        diagnostics/monitor output naming.
    metadata : dict
        Free-form provenance (e.g. the originating Bmad key and index when
        created via :meth:`ImpactXInput.from_tao`).
    """

    _impactx_class_: ClassVar[str]

    name: str = ""
    metadata: dict[str, int | float | str | bool] = {}

    def __init_subclass__(cls, impactx_class: str, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls._impactx_class_ = impactx_class
        input_element_by_class[impactx_class] = cls

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
        from impactx import elements

        cls = getattr(elements, self._impactx_class_)
        return cls(**self._impactx_kwargs())


class _Aligned(InputElement, impactx_class="_Aligned"):
    """Mixin fields shared by thick, alignable elements."""

    dx: float = 0.0
    dy: float = 0.0
    rotation: float = 0.0
    aperture_x: float = 0.0
    aperture_y: float = 0.0
    nslice: int = 1


# Remove the helper mixin from the registry -- it is not a real element.
input_element_by_class.pop("_Aligned", None)


class Drift(_Aligned, impactx_class="Drift"):
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


class Quad(_Aligned, impactx_class="Quad"):
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


class Sbend(_Aligned, impactx_class="Sbend"):
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


class CFbend(_Aligned, impactx_class="CFbend"):
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


class Sol(_Aligned, impactx_class="Sol"):
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


class DipEdge(InputElement, impactx_class="DipEdge"):
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


class Multipole(InputElement, impactx_class="Multipole"):
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


class Aperture(InputElement, impactx_class="Aperture"):
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


class Marker(InputElement, impactx_class="Marker"):
    """A zero-length marker with no effect on the beam."""

    type: Literal["Marker"] = "Marker"

    def _impactx_kwargs(self) -> dict[str, Any]:
        return {"name": self.name or "marker"}


class BeamMonitor(InputElement, impactx_class="BeamMonitor"):
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


AnyInputElement: TypeAlias = Union[
    Drift,
    Quad,
    Sbend,
    CFbend,
    Sol,
    DipEdge,
    Multipole,
    Aperture,
    Marker,
    BeamMonitor,
]


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
        Enable coherent synchrotron radiation.
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
    slice_step_diagnostics: bool = True

    def to_impactx_lattice(self) -> list:
        """Return the list of live ``impactx.elements`` objects."""
        return [ele.to_impactx() for ele in self.lattice]

    @classmethod
    def from_tao(cls, tao, **kwargs: Any) -> "ImpactXInput":
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
