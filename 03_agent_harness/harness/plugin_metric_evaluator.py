"""Harness 插件配置 — 接入真实指标评估器。

用已训练好的 v2（冻结）或 v4b 指标作为 ExplorerAgent 的 evaluator，
让 harness 能跑真实的一致性评估（而不是演示的假数据）。

两个 evaluator：
- MetricEvaluator：固定用全部 64 维特征（兼容旧 harness_round_001/002）
- MaskedMetricEvaluator：按 combo 的族 mask 选特征 → 支持族级组合搜索

Stage 1 自动化请用 MaskedMetricEvaluator。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# 让 harness 能找到 code 包
_CODE = Path(r"E:\ai4s\poetry-poetricity\02_environment\baseline_metrics")
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from code import build_and_freeze, FrozenMetric, FEATURE_NAMES  # noqa: E402
from code.data_loader_v2 import load_v2, train_val_split  # noqa: E402

# 复用 harness 的 AccessViolation 定义（避免循环 import）
try:
    from harness.harness import AccessViolation
except ImportError:
    # 直接运行本文件时
    import sys as _s
    _s.path.insert(0, str(Path(__file__).resolve().parent))
    from harness import AccessViolation  # type: ignore  # noqa: E402

# 族映射
try:
    from harness.family_map import FEATURE_FAMILIES, build_family_mask, get_all_family_features
except ImportError:
    _s.path.insert(0, str(Path(__file__).resolve().parent))
    from family_map import FEATURE_FAMILIES, build_family_mask, get_all_family_features  # type: ignore  # noqa: E402

from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
import numpy as np


# ============================================================================
# 旧版：固定全特征（保持兼容）
# ============================================================================

class MetricEvaluator:
    """把已训练指标包装成 ExplorerAgent 的 evaluator。

    combo 参数兼容 harness 的 explore_combo 动作：
      combo = {"features": [...], "model_version": "v2"|"v4b"|...}

    注：当前实现忽略 combo['features']，只用全部 64 维特征。
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
        metric = self._get_metric()
        samples = load_v2()
        _, val = train_val_split(samples, val_ratio=0.2, seed=self.seed)
        if split == "val":
            texts = [s.text for s in val]
            labels = [s.label for s in val]
        else:
            raise AccessViolation(f"split={split} 不允许，测试集不可访问")

        preds = np.asarray([metric.apply(t).pred for t in texts])
        y = np.asarray(labels)
        return {
            "kappa": float(cohen_kappa_score(y, preds, weights="quadratic")),
            "accuracy": float(accuracy_score(y, preds)),
            "f1_macro": float(f1_score(y, preds, average="macro")),
            "n_val": int(len(y)),
            "model_version": self.model_version,
            "failures": [],
        }


# ============================================================================
# 新版：族级 mask 评估器（Stage 1 自动化用）
# ============================================================================

class MaskedMetricEvaluator:
    """族级 mask 评估器：按 combo['families'] 选特征，mask 其余。

    一次训练（全部 64 维 LR），eval 时对未启用族乘 0 → 不影响 LR 系数但
    让那些特征在预测中贡献为 0。

    trade-off：
    - 优势：<0.1 秒/组合 → 2^13 = 8192 组合几秒钟穷举
    - 局限：mask 不重训，LR 系数受"启用族"影响（系数被无关特征干扰）

    combo 格式：
      combo = {
          "families": ["meter", "lang", "jump", "music"],
          "iteration": int,         # 可选
          "weights": {...},         # 可选，目前未用（LR 自动学）
      }
    """

    def __init__(self,
                 model_version: str = "v2",
                 seed: int = 42,
                 cache_predictions: bool = True):
        """
        Args:
            model_version: "v2" 或 "v4b"（当前仅 v2）
            seed: 训练 + 数据切分种子
            cache_predictions: 是否缓存 val 文本的全特征预测（大幅加速）
        """
        self.model_version = model_version
        self.seed = seed
        self.cache_predictions = cache_predictions

        # 延迟初始化（首次 __call__ 才加载数据 + 训练）
        self._metric: FrozenMetric | None = None
        self._val_texts: list[str] | None = None
        self._val_labels: np.ndarray | None = None
        # 全特征矩阵 (n_val, 64) + scaler 变换
        self._X_val_scaled: np.ndarray | None = None
        # 缓存全特征概率预测（避免重复 apply）
        self._probs_all: np.ndarray | None = None

    def _initialize(self) -> None:
        """首次调用时初始化：加载数据 + 训练 frozen_metric + 缓存全特征。"""
        if self._metric is not None:
            return
        # 1) 加载并训练 frozen metric（一次）
        self._metric = build_and_freeze(seed=self.seed, val_ratio=0.2)

        # 2) 加载 val 文本 + 标签
        samples = load_v2()
        _, val = train_val_split(samples, val_ratio=0.2, seed=self.seed)
        self._val_texts = [s.text for s in val]
        self._val_labels = np.asarray([s.label for s in val], dtype=np.int64)

        # 3) 提取全部 64 维特征 + 5 维 baseline = 69 维 → scaler 变换
        # 注：frozen_metric.apply() 内部已做这件事
        # 这里直接用 scaler + clf 避免 64 维 + 5 baseline 的混淆
        from code import extract_batch
        X_full = extract_batch(self._val_texts)  # (n_val, 64)
        # baseline 特征（5 维）
        base_train_count = 5
        # 复用 metric 内部的 baseline 计算
        baseline_arr = np.zeros((len(self._val_texts), 5), dtype=np.float64)
        # 直接调用 frozen_metric 内部接口提取 baseline
        for i, t in enumerate(self._val_texts):
            base_dict = self._metric._base_feats(t)
            baseline_arr[i] = [
                base_dict["bleu_to_poem"],
                base_dict["bleu_to_nonpoem"],
                base_dict["bigram_jacc_to_poem"],
                base_dict["bigram_jacc_to_nonpoem"],
                base_dict["tfidf_cos_to_poem"],
            ]
        # baseline_arr 顺序必须与 _base_feats 一致
        X_full_69 = np.concatenate([X_full, baseline_arr], axis=1)
        # scaler 已经在 metric 里 fit 过，直接 transform
        X_scaled = self._metric.scaler.transform(X_full_69)
        self._X_val_scaled = X_scaled

        # 4) 缓存全特征概率预测（prob[:, 1] = prob_poem）
        self._probs_all = self._metric.clf.predict_proba(X_scaled)[:, 1]

    def _eval_masked(self, active_families: list[str]) -> dict:
        """对 active_families 构造 mask，预测，计算 kappa/acc/f1。"""
        self._initialize()
        if not active_families:
            # 全 mask：所有特征被屏蔽 → 等于 baseline-only（退化情况）
            mask_64 = [False] * 64
        else:
            mask_64 = build_family_mask(active_families)

        # 构造 69 维 mask（64 特征 + 5 baseline，baseline 始终启用）
        mask_69 = np.asarray(mask_64 + [True] * 5, dtype=bool)

        # 应用 mask
        X_masked = self._X_val_scaled[:, mask_69]

        # 重新训练一个 mini LR（mask 后）—— 比直接乘 0 更准确
        # 因为 LR 系数会重新适应"启用特征子集"的方差
        # 但 mini LR 比 frozen_metric 的全特征 LR 慢很多（每组合都训练）
        # 折中：用全特征 LR，但把未启用特征置 0
        # 注：因为 StandardScaler 之后特征已是单位方差，置 0 等价于忽略
        X_with_zero = self._X_val_scaled.copy()
        # 找到未启用特征的列索引（在 69 维中）
        # 因为 baseline 始终启用，置 0 的是 0..63 中未启用的
        for i in range(64):
            if not mask_64[i]:
                X_with_zero[:, i] = 0.0

        probs = self._metric.clf.predict_proba(X_with_zero)[:, 1]
        preds = (probs >= 0.5).astype(np.int64)
        y = self._val_labels

        kappa = float(cohen_kappa_score(y, preds, weights="quadratic"))
        acc = float(accuracy_score(y, preds))
        f1 = float(f1_score(y, preds, average="macro"))

        return {
            "kappa": kappa,
            "accuracy": acc,
            "f1_macro": f1,
            "n_val": int(len(y)),
            "model_version": self.model_version,
            "active_families": active_families,
            "n_features_active": int(sum(mask_64)),
            "failures": [],
        }

    def __call__(self, combo: dict, split: str = "val") -> dict:
        """评估 combo 在 val 上的一致性。

        Args:
            combo: 必须含 "families" key（list[str]）
            split: 仅 "val" 允许（测试集不可访问，方案 §2.1）
        """
        if split != "val":
            raise AccessViolation(f"split={split} 不允许，测试集不可访问")

        t0 = time.perf_counter()
        families = combo.get("families")
        if families is None:
            # 兼容旧 combo 格式 {"features": ["all"]}
            if "features" in combo and combo["features"] == ["all"]:
                families = list(FEATURE_FAMILIES.keys())
            else:
                families = list(FEATURE_FAMILIES.keys())  # 默认全启用

        result = self._eval_masked(families)
        result["eval_seconds"] = time.perf_counter() - t0
        return result


# ============================================================================
# 自测
# ============================================================================

if __name__ == "__main__":
    print("=== 真实评估器自测 ===\n")

    # 旧版
    ev = MetricEvaluator(model_version="v2")
    result = ev({"features": ["all"], "model_version": "v2"}, split="val")
    print(f"MetricEvaluator (全特征): kappa={result['kappa']:.4f} acc={result['accuracy']:.4f} "
          f"f1={result['f1_macro']:.4f} n={result['n_val']}")
    try:
        ev({"features": []}, split="test")
    except AccessViolation as e:
        print(f"  test split 被拒绝: {e}")

    print("\n=== MaskedMetricEvaluator 自测 ===\n")
    mev = MaskedMetricEvaluator()
    # 单族
    r = mev({"families": ["meter"]})
    print(f"仅 meter ({r['n_features_active']} 维): kappa={r['kappa']:.4f} "
          f"acc={r['accuracy']:.4f} ({r['eval_seconds']:.3f}s)")
    # 4 族（论文 Stage 1 v2 用过）
    r = mev({"families": ["meter", "lang", "jump", "music"]})
    print(f"4 族 ({r['n_features_active']} 维): kappa={r['kappa']:.4f} "
          f"acc={r['accuracy']:.4f} ({r['eval_seconds']:.3f}s)")
    # 全族
    r = mev({"families": list(FEATURE_FAMILIES.keys())})
    print(f"全 13 族 ({r['n_features_active']} 维): kappa={r['kappa']:.4f} "
          f"acc={r['accuracy']:.4f} ({r['eval_seconds']:.3f}s)")
    # 单族 meter + lang
    r = mev({"families": ["meter", "lang"]})
    print(f"仅 meter+lang ({r['n_features_active']} 维): kappa={r['kappa']:.4f} "
          f"acc={r['accuracy']:.4f}")
