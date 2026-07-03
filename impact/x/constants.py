from __future__ import annotations

import enum


class SpaceChargeSolver(str, enum.Enum):
    """Poisson solver used for the space-charge calculation."""

    none = "none"
    fft = "fft"
    multigrid = "multigrid"


class ApertureShape(str, enum.Enum):
    """Transverse aperture cross-section shape."""

    rectangular = "rectangular"
    elliptical = "elliptical"


class ApertureAction(str, enum.Enum):
    """What an aperture does to particles outside its bounds."""

    transmit = "transmit"
    absorb = "absorb"


class GradientUnit(enum.IntEnum):
    """Unit convention for chromatic/soft element field strengths.

    ImpactX elements that accept a ``unit`` flag use ``0`` for the
    dimensionless MAD-X-like convention and ``1`` for physical field units
    (e.g. ``T/m`` for a quadrupole) which require the reference rigidity.
    """

    dimensionless = 0
    physical = 1


class OpenPMDBackend(str, enum.Enum):
    """openPMD file backend for :class:`BeamMonitor` output."""

    default = "default"
    h5 = "h5"
    bp = "bp"
    bp4 = "bp4"
