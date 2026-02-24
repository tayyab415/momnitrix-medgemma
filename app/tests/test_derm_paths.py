import os
from pathlib import Path
from tempfile import TemporaryDirectory

from momnitrix.model_runtime import _resolve_derm_artifact_path


def test_resolve_derm_artifact_path_prefers_configured_path():
    with TemporaryDirectory() as tmp:
        configured = Path(tmp) / "custom_classifier.pkl"
        configured.write_bytes(b"x")

        resolved = _resolve_derm_artifact_path(str(configured), "derm_classifier.pkl")
        assert resolved == configured


def test_resolve_derm_artifact_path_falls_back_to_local_artifacts():
    with TemporaryDirectory() as tmp:
        previous_cwd = Path.cwd()
        os.chdir(tmp)
        try:
            local_path = Path("artifacts/derm/derm_classifier.pkl")
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(b"x")

            resolved = _resolve_derm_artifact_path(
                "/root/artifacts/derm/derm_classifier.pkl",
                "derm_classifier.pkl",
            )
            assert resolved == local_path
        finally:
            os.chdir(previous_cwd)
