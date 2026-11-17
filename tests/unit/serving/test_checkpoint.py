"""Seam: slim checkpoint contract and export CLI (FEAT-003 FR03-FR05, I5)."""

import hashlib

import numpy as np
import pytest
import torch

from src.legislabert import LegislaBERTClassifier
from src.serving.checkpoint import CheckpointError, load_slim_checkpoint
from src.serving.export import export_slim_checkpoint, main
from tests.fixtures.synthetic_statutes import MockBertBackbone


def _make_model(seed: int = 0) -> LegislaBERTClassifier:
    torch.manual_seed(seed)
    return LegislaBERTClassifier(backbone=MockBertBackbone(hidden_size=768))


def _factory() -> LegislaBERTClassifier:
    return _make_model(seed=123)  # different init so loading must overwrite it


def _training_checkpoint(path, model=None, max_length=512):
    """Mimic the dict written by run_fine_tuning (includes optimizer + numpy metrics)."""
    model = model or _make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    torch.save(
        {
            "epoch": 3,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": {
                "f1_macro": np.float64(0.734),
                "recall_veto": np.float64(0.5),
                "classification_report": {"1": {"recall": np.float64(0.5)}},
            },
            "config": {"model_name": "neuralmind/bert-base-portuguese-cased", "max_length": max_length},
        },
        path,
    )
    return model


def test_export_strips_optimizer_and_sanitizes_metrics(tmp_path):
    src, dst = tmp_path / "full.pt", tmp_path / "slim.pt"
    _training_checkpoint(src)

    export_slim_checkpoint(src, dst, target="teve_veto")

    slim = torch.load(dst, map_location="cpu", weights_only=True)  # must be loadable safely
    assert set(slim) == {"model_state_dict", "config", "metrics", "target", "sha256_of_source"}
    assert slim["target"] == "teve_veto"
    assert slim["sha256_of_source"] == hashlib.sha256(src.read_bytes()).hexdigest()
    assert type(slim["metrics"]["f1_macro"]) is float
    assert dst.stat().st_size < src.stat().st_size


def test_export_rejects_unknown_target(tmp_path):
    src = tmp_path / "full.pt"
    _training_checkpoint(src)
    with pytest.raises(ValueError, match="target"):
        export_slim_checkpoint(src, tmp_path / "slim.pt", target="nonsense")


def test_export_rejects_checkpoint_without_state_dict(tmp_path):
    src = tmp_path / "bad.pt"
    torch.save({"config": {"model_name": "x", "max_length": 512}}, src)
    with pytest.raises(CheckpointError, match="model_state_dict"):
        export_slim_checkpoint(src, tmp_path / "slim.pt", target="teve_veto")


def test_export_cli_main_writes_file(tmp_path):
    src, dst = tmp_path / "full.pt", tmp_path / "slim.pt"
    _training_checkpoint(src)
    code = main(["--src", str(src), "--dst", str(dst), "--target", "instavel"])
    assert code == 0 and dst.exists()


def test_round_trip_loads_identical_weights_in_eval_mode(tmp_path):
    original = _training_checkpoint(tmp_path / "full.pt")
    export_slim_checkpoint(tmp_path / "full.pt", tmp_path / "slim.pt", target="teve_veto")

    loaded = load_slim_checkpoint(tmp_path / "slim.pt", _factory, expected_target="teve_veto")

    assert loaded.target == "teve_veto"
    assert loaded.max_length == 512
    assert loaded.model.training is False
    assert loaded.sha256 == hashlib.sha256((tmp_path / "slim.pt").read_bytes()).hexdigest()
    for key, tensor in original.state_dict().items():
        assert torch.equal(loaded.model.state_dict()[key], tensor)


def test_load_without_expected_target_accepts_any(tmp_path):
    _training_checkpoint(tmp_path / "full.pt")
    export_slim_checkpoint(tmp_path / "full.pt", tmp_path / "slim.pt", target="instavel")
    assert load_slim_checkpoint(tmp_path / "slim.pt", _factory).target == "instavel"


def test_target_mismatch_aborts(tmp_path):
    _training_checkpoint(tmp_path / "full.pt")
    export_slim_checkpoint(tmp_path / "full.pt", tmp_path / "slim.pt", target="teve_veto")
    with pytest.raises(CheckpointError, match="target"):
        load_slim_checkpoint(tmp_path / "slim.pt", _factory, expected_target="instavel")


def test_missing_file_aborts(tmp_path):
    with pytest.raises(CheckpointError, match="not found"):
        load_slim_checkpoint(tmp_path / "absent.pt", _factory)


def test_wrong_max_length_aborts(tmp_path):
    _training_checkpoint(tmp_path / "full.pt", max_length=256)
    export_slim_checkpoint(tmp_path / "full.pt", tmp_path / "slim.pt", target="teve_veto")
    with pytest.raises(CheckpointError, match="max_length"):
        load_slim_checkpoint(tmp_path / "slim.pt", _factory)


def test_missing_keys_abort(tmp_path):
    torch.save({"model_state_dict": {}}, tmp_path / "slim.pt")
    with pytest.raises(CheckpointError, match="missing keys"):
        load_slim_checkpoint(tmp_path / "slim.pt", _factory)


def test_state_dict_mismatch_aborts(tmp_path):
    _training_checkpoint(tmp_path / "full.pt")
    export_slim_checkpoint(tmp_path / "full.pt", tmp_path / "slim.pt", target="teve_veto")
    slim = torch.load(tmp_path / "slim.pt", weights_only=True)
    slim["model_state_dict"].pop("classifier.bias")
    torch.save(slim, tmp_path / "broken.pt")
    with pytest.raises(CheckpointError, match="state_dict"):
        load_slim_checkpoint(tmp_path / "broken.pt", _factory)


def test_unreadable_file_aborts(tmp_path):
    (tmp_path / "junk.pt").write_bytes(b"not a torch file")
    with pytest.raises(CheckpointError, match="read"):
        load_slim_checkpoint(tmp_path / "junk.pt", _factory)
