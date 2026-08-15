"""Benchmark semantic batch with cache on real corpus texts."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(r"E:\ai4s\poetry-poetricity\02_environment\baseline_metrics")))
from code.data_loader_v2 import load_v2
from code.semantic import semantic_features_batch, _load_cache

print("cache size:", len(_load_cache()))
samples = load_v2()
texts = [s.text for s in samples]
print("n texts:", len(texts))

# first call (should hit cache, no model load)
t0 = time.time()
r = semantic_features_batch(texts[:50])
print(f"50 cached: {time.time()-t0:.3f}s")

t0 = time.time()
r = semantic_features_batch(texts[:200])
print(f"200 cached: {time.time()-t0:.3f}s")

t0 = time.time()
r = semantic_features_batch(texts)
print(f"all {len(texts)} cached: {time.time()-t0:.3f}s")