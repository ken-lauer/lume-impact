"""Shared pydantic types for the ImpactX (``impact.x``) support.

The base model machinery is identical to the ImpactZ implementation, so it is
re-exported here rather than duplicated.
"""

from __future__ import annotations

from ..z.types import (
    AnyPath,
    BaseModel,
    NDArray,
    NegativeFloat,
    NonzeroFloat,
    PositiveFloat,
    PydanticParticleGroup,
)

__all__ = [
    "AnyPath",
    "BaseModel",
    "NDArray",
    "NegativeFloat",
    "NonzeroFloat",
    "PositiveFloat",
    "PydanticParticleGroup",
]
