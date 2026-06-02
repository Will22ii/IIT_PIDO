from __future__ import annotations

import unittest

from Optimizer.config import split_optimizer_system_config
from pipeline.config_io import pipeline_config_from_dict


class PipelineOptimizerConfigContractTest(unittest.TestCase):
    def test_optimizer_system_flat_overrides_remain_supported(self) -> None:
        config = pipeline_config_from_dict(
            {
                "problem": {
                    "name": "rosenbrock_nodummy",
                    "seed": 123,
                    "objective_sense": "min",
                },
                "run": {
                    "debug_level": "off",
                    "use_timestamp": False,
                },
                "tasks": {
                    "doe": False,
                    "modeler": False,
                    "explorer": False,
                    "optimizer": True,
                },
                "optimizer": {
                    "n_samples": 7,
                    "goal": 1.2,
                    "system": {
                        "algorithm_id": "my_runtime_algorithm",
                        "algorithm_params": {"alpha": 0.25},
                        "focus_pipeline": "focus0,focus1,focus3",
                        "focus3_refine_starts": 11,
                    },
                },
            }
        )

        self.assertIsNotNone(config.optimizer)
        self.assertEqual(config.optimizer.user.n_samples, 7)
        self.assertEqual(config.optimizer.user.goal, 1.2)
        self.assertEqual(config.optimizer.system.algorithm_id, "my_runtime_algorithm")
        self.assertEqual(config.optimizer.system.algorithm_params, {"alpha": 0.25})
        self.assertEqual(config.optimizer.system.focus_pipeline, "focus0,focus1,focus3")
        self.assertEqual(config.optimizer.system.focus3_refine_starts, 11)

        split = split_optimizer_system_config(
            config.optimizer.system,
            algorithm_id=config.optimizer.system.algorithm_id,
        )
        self.assertEqual(split["algorithm_contract"], "optimizer_runtime")
        self.assertEqual(split["algorithm"], {"algorithm_params": {"alpha": 0.25}})
        self.assertIn("focus3_refine_starts", split["focus_bo"])


if __name__ == "__main__":
    unittest.main()
