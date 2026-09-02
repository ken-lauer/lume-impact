"""The :class:`ImpactX` driver.

ImpactX is a Python-driven library (compiled AMReX code with pybind11 bindings)
rather than a standalone executable, so this driver builds and runs the
simulation through the ImpactX Python API.

Two execution modes are supported:

* **subprocess** (default) -- the input is serialized and executed by a fresh
  ``python`` interpreter in the working directory.  This isolates AMReX's
  once-per-process global initialize/finalize, so many runs (e.g. a pytest
  suite) do not interfere with one another.
* **in-process** -- the simulation is built and tracked in the current
  interpreter.  Faster, but only safe for a single run per process.
"""

from __future__ import annotations

import contextlib
import os
import pathlib
import subprocess
import sys
import tempfile
from typing import TYPE_CHECKING, Any

from beamphysics import ParticleGroup
from beamphysics.species import charge_state, mass_of
from impactx import ImpactX as _ImpactXSim

from .input import ImpactXInput
from .output import ImpactXOutput
from .particles import particle_group_to_impactx

if TYPE_CHECKING:
    from pytao import Tao

INPUT_JSON = "impactx_input.json"
INITIAL_PARTICLES_H5 = "initial_particles.h5"
RUN_SCRIPT = "run_impactx.py"


@contextlib.contextmanager
def _chdir(path: pathlib.Path):
    prev = pathlib.Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def configure_reference_particle(
    ref,
    species: str,
    kin_energy_MeV: float,
    mass_MeV: float | None = None,
    charge_qe: float | None = None,
) -> None:
    """Configure an ImpactX reference particle, across ImpactX versions.

    Newer ImpactX exposes ``RefPart.set_species``; older releases require the
    mass and charge to be set explicitly, which are looked up from beamphysics.
    """
    if hasattr(ref, "set_species"):
        ref.set_species(species)
    if mass_MeV is not None:
        ref.set_mass_MeV(mass_MeV)
    elif not hasattr(ref, "set_species"):
        ref.set_mass_MeV(mass_of(species) / 1e6)
    if charge_qe is not None:
        ref.set_charge_qe(charge_qe)
    elif not hasattr(ref, "set_species"):
        ref.set_charge_qe(charge_state(species))
    ref.set_kin_energy_MeV(kin_energy_MeV)
    ref.z = 0.0


def build_and_track(input: ImpactXInput, initial_particles: ParticleGroup | None):
    """Build an ImpactX simulation from ``input`` and track it to completion.

    Runs in the current working directory; diagnostics are written under
    ``./diags``.  The caller is responsible for the working directory.
    """
    sim = _ImpactXSim()
    # particle_shape must be set before init_grids for collective effects.
    sim.particle_shape = input.particle_shape
    sim.space_charge = input.space_charge
    sim.csr = input.csr
    if input.csr:
        sim.csr_bins = input.csr_bins
    sim.slice_step_diagnostics = input.slice_step_diagnostics
    sim.init_grids()

    pc = sim.particle_container()
    ref = pc.ref_particle()
    configure_reference_particle(
        ref,
        species=input.species,
        kin_energy_MeV=input.kin_energy_MeV,
        mass_MeV=input.mass_MeV,
        charge_qe=input.charge_qe,
    )

    if initial_particles is not None:
        particle_group_to_impactx(
            pc, ref, initial_particles, input.bunch_charge_C or None
        )
    else:
        raise NotImplementedError(
            "Distribution-based beam initialization is not yet implemented; "
            "provide initial_particles."
        )

    sim.lattice.extend(input.to_impactx_lattice())
    sim.track_particles()
    sim.finalize()


def _run_from_files(input_json: str, particles_h5: str | None) -> None:
    """Entry point used by the generated subprocess run script."""
    input = ImpactXInput.model_validate_json(pathlib.Path(input_json).read_text())
    particles = (
        ParticleGroup(h5=particles_h5)
        if particles_h5 and pathlib.Path(particles_h5).exists()
        else None
    )
    build_and_track(input, particles)


class ImpactX:
    """Driver for an ImpactX simulation described by an :class:`ImpactXInput`.

    Parameters
    ----------
    input : ImpactXInput
        The simulation description.
    workdir : str or pathlib.Path, optional
        Base directory for the run.  A temporary subdirectory is created within
        it when ``use_temp_dir`` is True.
    use_temp_dir : bool
        Create a temporary working directory (default True).
    initial_particles : ParticleGroup, optional
        Overrides ``input.initial_particles``.
    """

    def __init__(
        self,
        input: ImpactXInput | None = None,
        *,
        workdir: str | pathlib.Path | None = None,
        use_temp_dir: bool = True,
        initial_particles: ParticleGroup | None = None,
        verbose: bool = False,
    ) -> None:
        self.input = input if input is not None else ImpactXInput()
        if initial_particles is not None:
            self.input.initial_particles = initial_particles
        self.workdir = pathlib.Path(workdir) if workdir is not None else None
        self.use_temp_dir = use_temp_dir
        self.verbose = verbose
        self.output: ImpactXOutput | None = None
        self.path: pathlib.Path | None = None

    @classmethod
    def from_tao(cls, tao: "Tao", **kwargs: Any) -> "ImpactX":
        """Create an :class:`ImpactX` from a live Tao instance's lattice.

        Keyword arguments are forwarded to :meth:`ImpactXInput.from_tao`.
        """
        input = ImpactXInput.from_tao(tao, **kwargs)
        return cls(input)

    def _resolve_workdir(self) -> pathlib.Path:
        base = self.workdir or pathlib.Path.cwd()
        base.mkdir(parents=True, exist_ok=True)
        if self.use_temp_dir:
            return pathlib.Path(tempfile.mkdtemp(prefix="impactx_", dir=base))
        return base

    def write_input(self, path: pathlib.Path) -> None:
        """Serialize the input (and initial particles) into ``path``."""
        path.mkdir(parents=True, exist_ok=True)
        (path / INPUT_JSON).write_text(self.input.model_dump_json())
        if self.input.initial_particles is not None:
            self.input.initial_particles.write(str(path / INITIAL_PARTICLES_H5))

    def run(self, in_process: bool = False) -> ImpactXOutput:
        """Run the simulation and parse its output.

        Parameters
        ----------
        in_process : bool
            Run in the current interpreter instead of a subprocess.  Only safe
            for a single run per process due to AMReX global state.
        """
        path = self._resolve_workdir()
        self.path = path
        if in_process:
            with _chdir(path):
                build_and_track(self.input, self.input.initial_particles)
        else:
            self.write_input(path)
            has_particles = self.input.initial_particles is not None
            particles_arg = repr(INITIAL_PARTICLES_H5) if has_particles else "None"
            script = (
                "from impact.x.run import _run_from_files\n"
                f"_run_from_files({INPUT_JSON!r}, {particles_arg})\n"
            )
            (path / RUN_SCRIPT).write_text(script)
            proc = subprocess.run(
                [sys.executable, RUN_SCRIPT],
                cwd=path,
                capture_output=not self.verbose,
                text=True,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"ImpactX run failed (exit {proc.returncode}).\n"
                    f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
                )
        self.output = ImpactXOutput.from_directory(path)
        return self.output
