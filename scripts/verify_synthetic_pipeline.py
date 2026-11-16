"""Exercise real Qlib/Alpha158/training/checkpoint/trading with synthetic prices.

This is an integration fixture, never evidence for the manuscript's IC/AR tables.
Run from repository root: python -m scripts.verify_synthetic_pipeline
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import ruamel.yaml as YAML


def write_market(root):
    root = Path(root)
    calendar = pd.bdate_range("2017-06-01", periods=120)
    symbols = [f"SH{600000 + i:06d}" for i in range(56)]
    for name in ("calendars", "instruments", "features"):
        (root / name).mkdir(parents=True, exist_ok=True)
    (root / "calendars/day.txt").write_text("\n".join(calendar.strftime("%Y-%m-%d")) + "\n", encoding="utf-8")
    for universe, codes in (("csi300", symbols), ("all", symbols + ["SH000300"])):
        (root / f"instruments/{universe}.txt").write_text("".join(
            f"{code}\t{calendar[0]:%Y-%m-%d}\t{calendar[-1]:%Y-%m-%d}\n" for code in codes), encoding="utf-8")
    rng = np.random.default_rng(2025)
    for i, code in enumerate(symbols + ["SH000300"]):
        change = .003*np.sin(np.arange(120)*.7+i*.2) + rng.normal(.0002, .006, 120)
        close = (10+i*.3)*np.cumprod(1+change)
        opening = close*(1+rng.normal(0, .001, 120))
        data = {"close": close, "open": opening, "high": np.maximum(close, opening)*1.01,
                "low": np.minimum(close, opening)*.99, "volume": rng.uniform(1e6, 2e6, 120),
                "vwap": (close+opening)/2, "factor": np.ones(120), "change": change}
        # Future fixture dates outside every evaluated split test exchange filters.
        if i == 54:
            data["change"][112] = .10
            data["close"][112] = data["close"][111]*1.10
        if i == 55:
            data["close"][112] = np.nan
        folder = root / "features" / code.lower()
        folder.mkdir(exist_ok=True)
        for name, values in data.items():
            np.r_[0., values].astype("<f4").tofile(folder / f"{name}.day.bin")
    return calendar


def run_verification(output):
    from qlib.utils import init_instance_by_config
    import qlib
    from src.run_utils import configure_run, portfolio_analysis
    from train import execute_backtest, evaluate_checkpoint, main
    from search import run_paper_selection

    torch.set_num_threads(2)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    dates = write_market(output / "market")
    config = YAML.YAML(typ="safe").load(Path("configs/f158_csi300/config_lstm.yaml").read_text(encoding="utf-8"))
    config["qlib_init"]["provider_uri"] = str(output / "market")
    handler = config["data_handler_config"]
    handler.update(start_time=str(dates[60].date()), end_time=str(dates[114].date()),
                   fit_start_time=str(dates[60].date()), fit_end_time=str(dates[89].date()))
    ds = config["task"]["dataset"]["kwargs"]
    ds["segments"] = {name: [str(dates[a].date()), str(dates[b].date())]
                      for name, a, b in (("train", 60, 89), ("valid", 90, 99), ("test", 100, 109))}
    ds["batch_size"] = 128
    config["model_config"].update(d_model=8, d_ff=8, e_layers=1, dropout=0.)
    config["task"]["model"]["kwargs"].update(n_epochs=1, plot_training=False)
    config["port_analysis_config"]["backtest"].update(
        start_time=str(dates[100].date()), end_time=str(dates[109].date()))
    qlib.init(provider_uri=str(output / "market"), region="cn", kernels=1)
    from qlib.backtest.exchange import Exchange
    from qlib.backtest.decision import Order
    from qlib.data import D
    exchange = Exchange(start_time=dates[111], end_time=dates[113], codes=["SH600000", "SH600054", "SH600055"],
                        **config["port_analysis_config"]["backtest"]["exchange_kwargs"])
    raw_price = D.features(["SH600000"], ["$close"], dates[112], dates[112]).iloc[0, 0]
    assert np.isclose(exchange.get_deal_price("SH600000", dates[112], dates[112], Order.BUY), raw_price*1.0001)
    assert np.isclose(exchange.get_deal_price("SH600000", dates[112], dates[112], Order.SELL), raw_price*.9999)
    assert not exchange.is_stock_tradable("SH600054", dates[112], dates[112])
    assert not exchange.is_stock_tradable("SH600055", dates[112], dates[112])
    rows, sample_indices = [], []
    for mode in ("original", "feature", "tsil"):
        cfg = copy.deepcopy(config)
        cfg["model_config"].update(injection_mode=mode, phi_type="mse" if mode == "original" else "phi_volatility")
        cfg["logdir"] = str(output / mode)
        cfg = configure_run(cfg, 42)
        dataset = init_instance_by_config(cfg["task"]["dataset"])
        # Independent raw-target oracle, through the actual Qlib expression loader.
        first = next(iter(dataset.prepare("valid")))
        indices = dataset.restore_index(first["index"])
        raw = D.features(list(indices.get_level_values("instrument").unique()), ["$close"],
                         dates[0], dates[-1])["$close"].unstack("instrument")
        expected = raw.shift(-1)/raw-1
        labels = np.array([expected.loc[date, stock] for stock, date in indices])
        np.testing.assert_allclose(first["label"].cpu().numpy(), labels, atol=2e-7)
        model = init_instance_by_config(cfg["task"]["model"])
        _, selected = model.fit(dataset, evaluate_test=False)
        checkpoint = Path(cfg["logdir"]) / "model.bin"
        preds, metrics = evaluate_checkpoint(model, dataset, checkpoint)
        assert len(preds) == 56 * 9, "split endpoint must be purged; every remaining stock-date retained"
        sample_indices.append(preds.index)
        test_data = dataset.prepare("test")
        test_data.eval()
        assert bool(next(iter(test_data))["predicates"]) == (mode == "feature")
        daily, positions = execute_backtest(preds, cfg["port_analysis_config"])
        assert daily["cost"].sum() > 0 and daily["turnover"].sum() > 0
        holdings = [amount for position in positions.values()
                    for stock in position.get_stock_list() for amount in [position.get_stock_amount(stock)]]
        assert holdings and all(abs(amount/100-round(amount/100)) < 1e-7 for amount in holdings)
        daily.to_csv(Path(cfg["logdir"]) / "backtest_daily.csv")
        rows.append({"mode": mode, "samples": len(preds), "best_valid_ic": selected["best_score"],
                     "test_ic": metrics["IC"], "trade_days": len(daily), "cost_total": daily["cost"].sum(),
                     "lot_rounding_verified": True, "portfolio": portfolio_analysis(daily)})
    assert all(index.equals(sample_indices[0]) for index in sample_indices)

    # Execute the actual search path with one frozen network and a tiny grid.
    config["model_config"]["injection_mode"] = "tsil"
    manifest = run_paper_selection(config, ["LSTM"], ["phi_momentum"], [-4., 4.], "csi300",
                                  output / "selection", seeds=[42],
                                  network_candidates=[{"lr": .001, "layers": 1, "d_model": 8, "bs": 128}])
    chosen = manifest["models"]["LSTM"]["selected_runs"][0]
    assert manifest["test_evaluated"] is False and Path(chosen["checkpoint"]).is_file()
    selected_config = YAML.YAML(typ="safe").load(Path(chosen["config_path"]).read_text(encoding="utf-8"))
    main(chosen["seed"], selected_config, chosen["config_path"], chosen["checkpoint"])
    assert (Path(chosen["checkpoint"]).parent / "test_evaluation/metrics.json").is_file()
    result = {"synthetic_only": True, "modes": rows, "equal_sample_indices": True,
              "selected_alpha": chosen["penalty_alpha"], "selection_test_evaluated": False,
              "raw_label_verified": True, "slippage_prices_verified": True,
              "suspension_and_threshold_filters_verified": True, "checkpoint_cli_verified": True}
    (output / "verification.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="verification_output/synthetic")
    args = parser.parse_args()
    print(json.dumps(run_verification(args.output), indent=2))
