"""一键运行 Stage 1 自动化指标组合探索。

用法：
    cd E:\\ai4s\\poetry-poetricity\\03_agent_harness\\harness
    python run_stage1_automated.py                    # 默认 200 轮 / greedy / patience 30
    python run_stage1_automated.py 500                # 跑 500 轮
    python run_stage1_automated.py 200 greedy 20      # 200 轮 + greedy + patience 20
    python run_stage1_automated.py 8192 exhaust 8192  # 穷举 2^13 = 8192 组合（~10-15 分钟）

产物：
    04_memory/experiment_logs/harness_round_NNN.json  (每轮 RoundRecord)
    03_agent_harness/harness/best_combo.json          (当前最佳 combo)
    03_agent_harness/harness/checkpoint.json          (每 10 轮完整状态)
    03_agent_harness/harness/run.log                  (人类可读进度日志)

结果：
    返回 (best_combo, best_kappa) — best_combo 是族组合 + iteration，best_kappa 是 Quadratic Weighted Kappa
    注意：mask 评估的 kappa 低于 v4b 0.974（mask 不重训），但**族组合排序**有意义
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


_HARNESS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _HARNESS_DIR.parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stage 1 自动化指标组合探索",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--max-rounds", type=int, default=200,
                        help="最大轮次（每轮评估一个族组合）")
    parser.add_argument("--strategy", choices=["greedy", "random", "exhaust", "iter"],
                        default="greedy",
                        help="ExplorerAgent 搜索策略")
    parser.add_argument("--patience", type=int, default=30,
                        help="早停：连续 N 轮无改善则停")
    parser.add_argument("--min-improvement", type=float, default=0.001,
                        help="视为改善的最小 kappa 增量")
    parser.add_argument("--checkpoint-every", type=int, default=10,
                        help="每 N 轮保存一次 checkpoint")
    parser.add_argument("--silent", action="store_true",
                        help="不打印进度到 stdout（仍写 run.log）")
    args = parser.parse_args()

    print("=" * 60)
    print("Stage 1 自动化指标组合探索")
    print("=" * 60)
    print(f"max_rounds={args.max_rounds}  strategy={args.strategy}  "
          f"patience={args.patience}  min_improvement={args.min_improvement}")
    print()

    # 导入 harness 模块
    sys.path.insert(0, str(_HARNESS_DIR))
    from harness import default_harness
    from plugin_metric_evaluator import MaskedMetricEvaluator
    from family_map import FEATURE_FAMILIES, ALL_FAMILIES
    from run_automated import AutomatedHarnessRunner

    print(f"[1/3] 构建评估器（首次调用需 ~90s 加载数据）...")
    t0 = time.perf_counter()
    evaluator = MaskedMetricEvaluator()
    # 预热：先跑一次全族评估，让缓存生效
    warmup = evaluator({"families": ALL_FAMILIES})
    print(f"      预热完成 ({time.perf_counter() - t0:.1f}s): "
          f"全 13 族 kappa={warmup['kappa']:.4f} acc={warmup['accuracy']:.4f}")
    print()

    print(f"[2/3] 构建 harness（strategy={args.strategy}）...")
    h = default_harness(evaluator=evaluator, strategy=args.strategy,
                        family_map=FEATURE_FAMILIES)
    print(f"      4 子 Agent: {[p.name for p in h.plugins.values()]}")
    print()

    print(f"[3/3] 启动自动化 runner...")
    print(f"      日志: {_HARNESS_DIR / 'run.log'}")
    print(f"      best: {_HARNESS_DIR / 'best_combo.json'}")
    print(f"      checkpoint: {_HARNESS_DIR / 'checkpoint.json'}")
    print(f"      harness_round: {_PROJECT_ROOT / '04_memory' / 'experiment_logs' / 'harness_round_NNN.json'}")
    print()
    print("=" * 60)
    print("开始运行（Ctrl+C 可中断，已实现 checkpoint 恢复）")
    print("=" * 60)
    print()

    runner = AutomatedHarnessRunner(
        h,
        max_rounds=args.max_rounds,
        early_stop_patience=args.patience,
        min_improvement=args.min_improvement,
        checkpoint_every=args.checkpoint_every,
        stage="stage1",
        silent=args.silent,
    )
    best_combo, best_kappa = runner.run()

    print()
    print("=" * 60)
    print("最终结果")
    print("=" * 60)
    print(f"best_kappa = {best_kappa:.4f}")
    print(f"best_combo = {best_combo}")
    print(f"  - 启用的族: {best_combo.get('families', [])}")
    print(f"  - 启用的族数: {len(best_combo.get('families', []))}/{len(ALL_FAMILIES)}")
    print()
    print("=" * 60)
    print("下一步建议")
    print("=" * 60)
    print("1. 用 best_combo 的族组合重新训练 LR（不 mask），可能 kappa > 0.974")
    print("2. 跨数据集验证 best combo 在 expert 集 + AI 诗集上的稳定性")
    print("3. 跑 v6 ablation 看 '哪些族是关键族'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
