"""Parsing of ImpactX run output.

Two products are read back:

* ``reduced_beam_characteristics`` -- a whitespace-delimited table of beam moments
  vs. the integrated path length ``s`` (the per-slice stats analog of ImpactZ's
  ``fort.*`` statistics files).
* ``diags/openPMD/*.h5`` -- per-:class:`BeamMonitor` particle dumps in the
  openPMD-beamphysics format, read directly into :class:`ParticleGroup`.
"""

from __future__ import annotations

import pathlib
from collections.abc import Mapping
from typing import Any

import numpy as np
from beamphysics import ParticleGroup


def _find_reduced_file(workdir: pathlib.Path) -> pathlib.Path | None:
    for pattern in (
        "reduced_beam_characteristics",
        "reduced_beam_characteristics*",
        "diags/reduced_beam_characteristics*",
    ):
        matches = sorted(workdir.glob(pattern))
        matches = [m for m in matches if m.is_file() and "final" not in m.name]
        if matches:
            return matches[0]
    return None


def load_reduced_beam_characteristics(path: pathlib.Path) -> dict[str, np.ndarray]:
    """Read a ``reduced_beam_characteristics`` file into a dict of columns."""
    with open(path) as fp:
        header = fp.readline().split()
    data = np.loadtxt(path, skiprows=1, ndmin=2)
    return {name: data[:, i] for i, name in enumerate(header)}


class ImpactXStats(Mapping):
    """Beam-moment statistics vs. ``s`` from ``reduced_beam_characteristics``.

    Columns are exposed both by mapping access (``stats["mean_x"]``) and as
    attributes (``stats.mean_x``).  Includes ``s``, ``mean_x``/``mean_y``,
    ``sigma_*``, ``emittance_*``, Twiss ``alpha_*``/``beta_*``, etc.
    """

    def __init__(self, columns: dict[str, np.ndarray]) -> None:
        self._columns = columns

    def __getitem__(self, key: str) -> np.ndarray:
        return self._columns[key]

    def __getattr__(self, key: str) -> np.ndarray:
        try:
            return self._columns[key]
        except KeyError as ex:
            raise AttributeError(key) from ex

    def __iter__(self):
        return iter(self._columns)

    def __len__(self) -> int:
        return len(self._columns)

    def __repr__(self) -> str:
        n = len(self._columns.get("s", []))
        return f"ImpactXStats(n={n}, keys={list(self._columns)})"


class ImpactXOutput:
    """Container for parsed ImpactX output.

    Attributes
    ----------
    stats : ImpactXStats
        Beam moments vs. ``s``.
    particles : dict
        Beam-monitor particle dumps keyed by monitor name.
    workdir : pathlib.Path
        Directory the output was read from.
    """

    def __init__(
        self,
        stats: ImpactXStats,
        particles: dict[str, Any],
        workdir: pathlib.Path,
    ) -> None:
        self.stats = stats
        self.particles = particles
        self.workdir = workdir

    @classmethod
    def from_directory(cls, workdir: str | pathlib.Path) -> "ImpactXOutput":
        workdir = pathlib.Path(workdir)
        reduced = _find_reduced_file(workdir)
        if reduced is None:
            raise FileNotFoundError(
                f"No reduced_beam_characteristics file found under {workdir}"
            )
        stats = ImpactXStats(load_reduced_beam_characteristics(reduced))
        particles = _load_monitor_particle_groups(workdir)
        return cls(stats=stats, particles=particles, workdir=workdir)

    def __repr__(self) -> str:
        return (
            f"ImpactXOutput(stats={self.stats!r}, "
            f"particles={list(self.particles)}, workdir={self.workdir})"
        )


def _load_monitor_particle_groups(workdir: pathlib.Path) -> dict[str, ParticleGroup]:
    """Read beam monitors, one ParticleGroup per (monitor, iteration).

    ImpactX writes particles in its own fixed-``s`` frame with reference-normalized
    momenta, which beamphysics cannot interpret directly.  Reading these back into
    lab-frame :class:`ParticleGroup` objects requires the inverse coordinate
    transform (not yet implemented); until then, monitors that cannot be parsed
    are skipped.  The per-``s`` ``stats`` table remains the primary output.
    """
    import warnings

    import h5py
    from beamphysics.readers import particle_paths

    diag_dir = workdir / "diags" / "openPMD"
    result: dict[str, ParticleGroup] = {}
    if not diag_dir.is_dir():
        return result

    for h5path in sorted(diag_dir.glob("*.h5")):
        with h5py.File(h5path, "r") as fp:
            try:
                paths = particle_paths(fp)
            except Exception:
                continue
            for path in paths:
                iteration = path.strip("/").split("/")[1]
                for species in fp[path]:
                    key = f"{h5path.stem}@{iteration}"
                    try:
                        result[key] = ParticleGroup(h5=fp[path][species])
                    except Exception as ex:  # noqa: BLE001
                        warnings.warn(
                            f"Could not read monitor {key} from {h5path.name}: {ex}. "
                            "ImpactX-frame particle read-back is not yet implemented; "
                            "use the reduced-beam stats table instead.",
                            stacklevel=2,
                        )
    return result
