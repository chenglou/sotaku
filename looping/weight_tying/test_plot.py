"""Check report rendering with synthetic data, never with the locked test set."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

from checkpoint_utils import atomic_json_save
from looping.weight_tying.common import protocol, run_name

HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None
if HAS_MATPLOTLIB:
    from PIL import Image
    from looping.weight_tying.plot import render_report


def rendering_fixture(directory):
    settings = protocol()
    runs, histories = {}, {}
    for architecture in settings["architectures"]:
        for regime in settings["regimes"]:
            for index, seed in enumerate(settings["seeds"]):
                name = run_name(architecture, regime, seed)
                scores = {str(horizon): {"accuracy": 0.90 + index * 0.02 - offset * 0.015}
                          for offset, horizon in enumerate(settings["evaluation"]["iterations"])}
                runs[name] = {"evaluations": {"final": {"holdout": scores, "development": scores}}}
                histories[name] = [{"updates": step, "scores": {
                    "16": {"accuracy": step / 22000}, "1024": {"accuracy": step / 21000}}}
                    for step in range(1000, 20001, 1000)]
    atomic_json_save({"partial": True, "synthetic": True, "runs": runs}, directory / "report.json")
    atomic_json_save(histories, directory / "learning_curves.json")


@unittest.skipUnless(HAS_MATPLOTLIB, "Matplotlib is only needed to render figures")
class PlotTests(unittest.TestCase):
    def test_renders_all_figures_without_overwriting(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            rendering_fixture(directory)
            paths = render_report(directory)
            self.assertEqual(len(paths), 3)
            for path in paths:
                with Image.open(path) as image:
                    self.assertGreaterEqual(image.width, 1900)
                    self.assertGreaterEqual(image.height, 800)
                    self.assertLess(image.convert("L").getextrema()[0], 100)
            with self.assertRaises(FileExistsError):
                render_report(directory)
