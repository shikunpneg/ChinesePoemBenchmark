"""Benchmark per-feature-family extraction time to find bottleneck."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(r"E:\ai4s\poetry-poetricity\02_environment\baseline_metrics")))

from code.meter import meter_to_features
from code.structure import structure_v2_features
from code.imagery_ner import imagery_features
from code.phonetics import phonetic_features
from code import feat_form, feat_structure, feat_logic_jump, feat_language, feat_purity, feat_style, feat_music_simple

sample = "床前明月光\n疑是地上霜\n举头望明月\n低头思故乡"

tests = {
    "meter": lambda: meter_to_features(sample),
    "structure_v2": lambda: structure_v2_features(sample),
    "imagery_ner": lambda: imagery_features(sample),
    "phonetics": lambda: phonetic_features(sample),
    "form": lambda: feat_form(sample),
    "struct": lambda: feat_structure(sample),
    "jump": lambda: feat_logic_jump(sample),
    "lang": lambda: feat_language(sample),
    "purity": lambda: feat_purity(sample),
    "style": lambda: feat_style(sample),
    "music": lambda: feat_music_simple(sample),
}

print("=== per-family timing (1 text) ===")
total = 0
for name, fn in tests.items():
    t0 = time.time()
    fn()
    dt = time.time() - t0
    total += dt
    print(f"  {name:14s}  {dt*1000:8.1f} ms")

print(f"\n  TOTAL (non-semantic): {total*1000:.1f} ms/text")
print(f"  x1516 train texts: {total*1516/60:.1f} min")
print(f"  x1855 all: {total*1855/60:.1f} min")