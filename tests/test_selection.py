import importlib.util
import ast
import json
from pathlib import Path

import pytest
import yaml


@pytest.mark.parametrize("source", ["lstm", "gru"])
def test_search_can_construct_every_target_backbone_from_cli_template(source):
    import torch
    from search import _apply
    from src.model_backbone_phitp import QniverseModel
    from src.run_utils import ALL_MODELS, RELATIONAL_MODELS, configure_run
    path = Path(__file__).parents[1] / f"configs/f158_csi300/config_{source}.yaml"
    base = yaml.safe_load(path.read_text(encoding="utf-8"))
    for name in ALL_MODELS:
        hp = {"d_model": 8, "layers": 1}
        hp["days_per_batch" if name in RELATIONAL_MODELS else "bs"] = 2 if name in RELATIONAL_MODELS else 128
        cfg = _apply(base, hp, name, "mse", 0., "csi300")
        cfg = configure_run(cfg, 42)
        model = QniverseModel(cfg["model_config"], model_type=name, seed=42, plot_training=False)
        model.model.eval()
        assert model._forward_batch({"data": torch.randn(3, 20, 159),
                                     "date": torch.zeros(3, dtype=torch.long)}).shape == (3,)


def test_relation_network_search_uses_day_batches():
    from search import _apply, network_search_space
    from src.run_utils import configure_run
    path = Path(__file__).parents[1] / "configs/f158_csi300/config_lstm.yaml"
    base = yaml.safe_load(path.read_text(encoding="utf-8"))
    space = network_search_space("GAT")
    assert "bs" not in space and space["days_per_batch"]["choices"] == [2, 4, 8]
    cfg = configure_run(_apply(base, {"days_per_batch": 8}, "GAT", "mse", 0., "csi300"), 42)
    assert cfg["task"]["dataset"]["kwargs"]["days_per_batch"] == 8
    assert network_search_space("LSTM")["bs"]["choices"] == [256, 512, 1024]


def test_feature_reuses_the_exact_original_network_selection(tmp_path):
    from search import run_paper_selection
    path = Path(__file__).parents[1] / "configs/f158_csi300/config_lstm.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg["model_config"]["injection_mode"] = "feature"
    frozen = {"market": "csi300", "models": {"LSTM": {"network_hyperparameters": {"d_model": 16},
              "network_selection": {"valid_ic": .3, "seed": 2025, "hyperparameters": {"d_model": 16}}}}}
    seen = []
    def train(config, name):
        assert config["model_config"]["injection_mode"] == "feature"
        assert config["model_config"]["d_model"] == 16
        seen.append(config["task"]["model"]["kwargs"]["seed"])
        return .1
    result = run_paper_selection(cfg, ["LSTM"], ["phi_cycle"], [-4., 4.], "csi300", tmp_path,
                                seeds=[42, 2022], train_fn=train, frozen_network_manifest=frozen)
    assert seen == [42, 2022]  # no network search, no meaningless beta sweep
    assert result["models"]["LSTM"]["network_selection"] == frozen["models"]["LSTM"]["network_selection"]


def test_network_frozen_before_seed_predicate_beta_selection(tmp_path):
    path = Path(__file__).parents[1] / "search.py"
    assert any(isinstance(n, ast.FunctionDef) and n.name == "run_paper_selection"
               for n in ast.parse(path.read_text(encoding="utf-8")).body), "paper's staged selection is missing"
    # Load the real entry point once dependencies are available.
    spec = importlib.util.spec_from_file_location("tsil_search", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module, "run_paper_selection"), "paper's staged selection is missing"
    cfg = yaml.safe_load((path.parent / "configs/f158_csi300/config_lstm.yaml").read_text(encoding="utf-8"))
    seen = []

    def train(config, model_type):
        # An independent score table is enough to verify orchestration; numerical
        # training is covered by the real trainer tests, not mocked here.
        mc = config["model_config"]
        seed = config["task"]["model"]["kwargs"]["seed"]
        seen.append((mc["phi_type"], seed, mc["d_model"], mc.get("penalty_alpha")))
        if mc["phi_type"] == "mse":
            assert seed == 2025
            return 0.2 if mc["d_model"] == 16 else 0.1
        assert mc["d_model"] == 16  # frozen for all seed/predicate/alpha combinations
        # Different seeds prefer different alphas; ones must never win main-table selection.
        if mc["phi_type"] == "ones":
            return 0.99
        optimal = -2.0 if seed == 42 else 2.0
        return (0.5 if mc["phi_type"] == "phi_cycle" else 0.3) - 0.1 * abs(mc["penalty_alpha"] - optimal)

    network_points = [{"d_model": 8}, {"d_model": 16}]
    manifest = module.run_paper_selection(cfg, ["LSTM"], ["phi_cycle", "phi_momentum", "ones"],
        [-2., 2.], "csi300", str(tmp_path), seeds=[42, 2025],
        network_candidates=network_points, train_fn=train)
    model = manifest["models"]["LSTM"]
    assert model["selected_predicate"] == "phi_cycle"
    assert model["network_hyperparameters"] == {"d_model": 16}
    assert {item["seed"]: item["penalty_alpha"] for item in model["selected_runs"]} == {42: -2., 2025: 2.}
    assert seen[:2] == [("mse", 2025, 8, 0.), ("mse", 2025, 16, 0.)]
    assert len(seen) == 2 + 3 * 2 * 2
    stored = json.loads((tmp_path / "selection_manifest.json").read_text(encoding="utf-8"))
    assert stored["test_evaluated"] is False
    assert "mean validation daily IC across seeds" in stored["predicate_selection_rule"]


def test_selection_refuses_all_failed_candidates(tmp_path):
    path = Path(__file__).parents[1] / "search.py"
    assert any(isinstance(n, ast.FunctionDef) and n.name == "run_paper_selection"
               for n in ast.parse(path.read_text(encoding="utf-8")).body), "paper's staged selection is missing"
    spec = importlib.util.spec_from_file_location("tsil_search", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module, "run_paper_selection")
    cfg = yaml.safe_load((path.parent / "configs/f158_csi300/config_lstm.yaml").read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="finite"):
        module.run_paper_selection(cfg, ["LSTM"], ["phi_cycle"], [0.], "csi300", str(tmp_path),
            seeds=[42], network_candidates=[{}], train_fn=lambda *args: float("nan"))
