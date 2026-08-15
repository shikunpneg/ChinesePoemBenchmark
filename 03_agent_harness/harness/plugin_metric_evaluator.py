"""Harness 插件配置 — 接入真实指标评估器。

用已训练好的 v2（冻结）或 v4b 指标作为 ExplorerAgent 的 evaluator，
让 harness 能跑真实的一致性评估（而不是演示的假数据）。
"""
from __future__ import annotations

import sys
from pathlib import Path

# 让 harness 能找到 code 包
_CODE = Path(r"E:\ai4s\poetry-poetricity\02_environment\baseline_metrics")
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from code import build_and_freeze, FrozenMetric  # noqa: E402
from code.data_loader_v2 import load_v2, train_val_split  # noqa: E402


class MetricEvaluator:
    """把已训练指标包装成 ExplorerAgent 的 evaluator。

    combo 参数兼容 harness 的 explore_combo 动作：
      combo = {"features": [...], "model_version": "v2"|"v4b"|...}
    """

    def __init__(self, model_version: str = "v2", seed: int = 42):
        self.model_version = model_version
        self.seed = seed
        self._metric = None

    def _get_metric(self):
        if self._metric is None:
            if self.model_version == "v2":
                self._metric = build_and_freeze(seed=self.seed, val_ratio=0.2)
            else:
                raise ValueError(f"model_version={self.model_version} 暂不支持，"
                                 "可用 v2")
        return self._metric

    def __call__(self, combo: dict, split: str = "val") -> dict:
        """评估一个指标组合在指定 split 上的人类一致性。"""
        metric = self._get_metric()
        samples = load_v2()
        _, val = train_val_split(samples, val_ratio=0.2, seed=self.seed)
        if split == "val":
            texts = [s.text for s in val]
            labels = [s.label for s in val]
        else:
            # 只允许 val（测试集不可见，方案 §2.1）
            raise AccessViolation(f"split={split} 不允许，测试集不可访问")

        from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
        import numpy as np

        preds = np.asarray([metric.apply(t).pred for t in texts])
        y = np.asarray(labels)
        return {
            "kappa": float(cohen_kappa_score(y, preds, weights="quadratic")),
            "accuracy": float(accuracy_score(y, preds)),
            "f1_macro": float(f1_score(y, preds, average="macro")),
            "n_val": int(len(y)),
            "model_version": self.model_version,
            "failures": [],  # 需要时从 round log 提取
        }


# 复用 harness 的 AccessViolation 定义（避免循环 import）
try:
    from harness.harness import AccessViolation
except ImportError:
    # 直接运行本文件时
    import sys as _s
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from harness import AccessViolation


if __name__ == "__main__":
    print("=== 真实评估器自测 ===")
    ev = MetricEvaluator(model_version="v2")
    result = ev({"features": ["all"], "model_version": "v2"}, split="val")
    print(f"v2 on val: kappa={result['kappa']:.4f} acc={result['accuracy']:.4f} "
          f"f1={result['f1_macro']:.4f} n={result['n_val']}")
    # 测试越界 split
    try:
        ev({"features": []}, split="test")
    except AccessViolation as e:
        print(f"  test split 被拒绝: {e}")