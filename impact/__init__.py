from .impact import Impact
from .impact_distgen import run_impact_with_distgen, evaluate_impact_with_distgen
from .control import ControlGroup
from .z import ImpactZ, ImpactZInput

try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.0"


def __getattr__(name: str):
    # impact.x requires the compiled `impactx` library; import it lazily so that
    # `import impact` works without it while `import impact.x` fails loudly.
    if name in ("ImpactX", "ImpactXInput"):
        from . import x

        return getattr(x, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Impact",
    "ImpactX",
    "ImpactXInput",
    "ImpactZ",
    "ImpactZInput",
    "run_impact_with_distgen",
    "evaluate_impact_with_distgen",
    "ControlGroup",
]
