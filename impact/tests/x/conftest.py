import logging
import pathlib

import pytest

x_tests = pathlib.Path(__file__).resolve().parent

test_artifacts = x_tests / "artifacts"
test_failure_artifacts = test_artifacts / "failures"

bmad_files = x_tests / "bmad"

logging.getLogger("pytao.subproc").setLevel("WARNING")
logging.getLogger("matplotlib.font_manager").setLevel("WARNING")


@pytest.fixture(autouse=True, scope="session")
def _make_artifacts_dir() -> None:
    test_failure_artifacts.mkdir(exist_ok=True, parents=True)
