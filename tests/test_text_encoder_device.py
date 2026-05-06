from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import v3_1_blue as model_module


def test_configure_model_runtime_dirs_defaults_to_project_workspace():
    env = {}
    paths = model_module.configure_model_runtime_dirs(environ=env, create_dirs=False)
    project_root = str(PROJECT_ROOT)

    assert paths["cache_root"].startswith(project_root)
    assert paths["temp_root"].startswith(project_root)
    assert env["HF_HOME"].startswith(project_root)
    assert env["SENTENCE_TRANSFORMERS_HOME"].startswith(project_root)
    assert env["TORCH_HOME"].startswith(project_root)
    assert env["TEMP"].startswith(project_root)
    assert env["TMP"].startswith(project_root)


def test_resolve_text_encoder_device_auto_prefers_cuda_when_available():
    assert model_module.resolve_text_encoder_device(
        "auto",
        cuda_available=lambda: True,
        cuda_device_count=lambda: 1,
    ) == "cuda:0"


def test_resolve_text_encoder_device_auto_uses_cpu_without_cuda():
    assert model_module.resolve_text_encoder_device(
        "auto",
        cuda_available=lambda: False,
        cuda_device_count=lambda: 0,
    ) == "cpu"


def test_resolve_text_encoder_device_honors_cpu_override():
    assert model_module.resolve_text_encoder_device(
        "cpu",
        cuda_available=lambda: True,
        cuda_device_count=lambda: 1,
    ) == "cpu"


def test_resolve_text_encoder_device_rejects_unavailable_cuda():
    try:
        model_module.resolve_text_encoder_device(
            "cuda",
            cuda_available=lambda: False,
            cuda_device_count=lambda: 0,
        )
    except RuntimeError as exc:
        assert "HR_TEXT_ENCODER_DEVICE=cuda" in str(exc)
    else:
        raise AssertionError("Expected unavailable CUDA override to fail clearly")


if __name__ == "__main__":
    test_configure_model_runtime_dirs_defaults_to_project_workspace()
    test_resolve_text_encoder_device_auto_prefers_cuda_when_available()
    test_resolve_text_encoder_device_auto_uses_cpu_without_cuda()
    test_resolve_text_encoder_device_honors_cpu_override()
    test_resolve_text_encoder_device_rejects_unavailable_cuda()
