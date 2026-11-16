"""Executable numerical regression that also runs against the released loss."""
import ast
import importlib.util
from pathlib import Path
import unittest

import torch


class PaperLossRegression(unittest.TestCase):
    def test_ones_paper_value(self):
        pred = torch.tensor([1., 2.])
        target = torch.zeros(2)
        if importlib.util.find_spec("src.tsil_loss"):
            from src.tsil_loss import StatisticalInvariantLoss
            result = StatisticalInvariantLoss(0.5)(pred, target, {"ones": torch.ones(2, 1)})
        else:
            # Isolate the *actual* old loss from unrelated optional backbone imports.
            path = Path(__file__).parents[1] / "src/model_backbone_phitp.py"
            tree = ast.parse(path.read_text(encoding="utf-8"))
            node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "WeightedMSELoss")
            namespace = {"torch": torch, "nn": torch.nn, "TEMPORAL_PHI_TYPES": set(),
                         "_CUSTOM_PHI_AVAILABLE": False}
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
            result = namespace["WeightedMSELoss"]()("ones", None, None, pred, target, 0.5, 0.5)
        self.assertAlmostEqual(result["total"].item(), 3.0625, places=7)


if __name__ == "__main__":
    unittest.main()
