"""Benchmark each family on 100 real texts to find the slow one."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(r"E:\ai4s\poetry-poetricity\02_environment\baseline_metrics")))
from code.data_loader_v2 import load_v2
from code.meter import meter_to_features
from code.structure import structure_v2_features
from code.imagery_ner import imagery_features
from code.phonetics import phonetic_features
from code import feat_form, feat_structure, feat_logic_jump, feat_language, feat_purity, feat_style, feat_music_simple

samples = load_v2()
texts = [s.text for s in samples[:100]]
print(f"benchmarking on {len(texts)} texts\n")

tests = {
    "meter": lambda ts: [meter_to_features(t) for t in ts],
    "structure_v2": lambda ts: [structure_v2_features(t) for t in ts],
    "imagery_ner": lambda ts: [imagery_features(t) for t in ts],
    "phonetics": lambda ts: [phonetic_features(t) for t in ts],
    "form": lambda ts: [feat_form(t) for t in ts],
    "struct": lambda ts: [feat_structure(t) for t in ts],
    "jump": lambda ts: [feat_logic_jump(t) for t in ts],
    "lang": lambda ts: [feat_language(t) for t in ts],
    "purity": lambda ts: [feat_purity(t) for t in ts],
    "style": lambda ts: [feat_style(t) for t in ts],
    "music": lambda ts: [feat_music_simple(t) for t in ts],
}
for name, fn in tests.items():
    t0 = time.time()
    fn(texts)
    dt = time.time() - t0
    print(f"  {name:14s}  {dt:6.2f}s for 100 texts  ({dt/100*1000:6.1f} ms/text)")