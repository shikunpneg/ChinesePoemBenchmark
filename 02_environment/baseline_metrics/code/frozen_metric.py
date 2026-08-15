"""Frozen Stage-1 metric.

This module encapsulates the logistic-regression model trained on the
Round-2 / Round-3 dataset (1855 samples = 1500 poems + 355 hard
nonpoems). The metric is "frozen": its feature pipeline, scaler, LR
weights, and baseline corpus are saved as a single pickle so that
Stage 2 can apply it to AI-generated poems WITHOUT retraining.

API:
    FrozenMetric.load(path) -> FrozenMetric
    fm.apply(text) -> FrozenPrediction
    fm.apply_batch(texts) -> list[FrozenPrediction]
    fm.save(path)  -> None
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from code import (
    FEATURE_NAMES,
    bleu_self,
    bigram_jaccard_self,
    build_char_vocab,
    build_tfidf_centroid,
    char_tfidf_cosine_to_poetry_centroid,
    extract_batch,
)
from code.data_loader_v2 import (
    class_char_counts,
    class_centroid_text,
    load_v2,
    train_val_split,
)


@dataclass
class FrozenPrediction:
    text: str
    prob_poem: float         # probability of label=1
    pred: int                # 0 or 1 (threshold 0.5)
    feature_dict: dict[str, float]   # all feature values, for debugging


class FrozenMetric:
    """A trained and frozen indicator."""

    def __init__(self,
                 clf: LogisticRegression,
                 scaler: StandardScaler,
                 counts_poem, counts_nonpoem,
                 counts_poem_2gram, counts_nonpoem_2gram,
                 vocab, centroid_poem_tfidf, idf_arr,
                 train_settings: dict) -> None:
        self.clf = clf
        self.scaler = scaler
        self.counts_poem = counts_poem
        self.counts_nonpoem = counts_nonpoem
        self.counts_poem_2gram = counts_poem_2gram
        self.counts_nonpoem_2gram = counts_nonpoem_2gram
        self.vocab = vocab
        self.centroid_poem_tfidf = centroid_poem_tfidf
        self.idf_arr = idf_arr
        self.train_settings = train_settings
        self.feat_names = FEATURE_NAMES + [
            "base_bleu_to_poem", "base_bleu_to_nonpoem",
            "base_bigram_jacc_to_poem", "base_bigram_jacc_to_nonpoem",
            "base_tfidf_cos_to_poem",
        ]

    def _base_feats(self, text: str) -> dict[str, float]:
        total_poem = self.train_settings["total_poem"]
        total_nonpoem = self.train_settings["total_nonpoem"]
        return {
            "bleu_to_poem": bleu_self(text, self.counts_poem, total_poem),
            "bleu_to_nonpoem": bleu_self(text, self.counts_nonpoem, total_nonpoem),
            "bigram_jacc_to_poem": bigram_jaccard_self(text, self.counts_poem_2gram),
            "bigram_jacc_to_nonpoem": bigram_jaccard_self(text, self.counts_nonpoem_2gram),
            "tfidf_cos_to_poem": char_tfidf_cosine_to_poetry_centroid(
                text, self.vocab, self.centroid_poem_tfidf, self.idf_arr),
        }

    def _featurize(self, text: str) -> tuple[np.ndarray, dict[str, float]]:
        feats = extract_batch([text])[0]
        base = np.asarray(
            list(self._base_feats(text).values()), dtype=np.float64)
        full = np.concatenate([feats, base])[None, :]
        names = self.feat_names
        d = {n: float(v) for n, v in zip(names, full[0])}
        return full, d

    def apply(self, text: str) -> FrozenPrediction:
        X, d = self._featurize(text)
        X_s = self.scaler.transform(X)
        prob = float(self.clf.predict_proba(X_s)[0, 1])
        pred = int(prob >= 0.5)
        return FrozenPrediction(text=text, prob_poem=prob, pred=pred,
                                feature_dict=d)

    def apply_batch(self, texts: Iterable[str]) -> list[FrozenPrediction]:
        return [self.apply(t) for t in texts]

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({
                "clf": self.clf,
                "scaler": self.scaler,
                "counts_poem": self.counts_poem,
                "counts_nonpoem": self.counts_nonpoem,
                "counts_poem_2gram": self.counts_poem_2gram,
                "counts_nonpoem_2gram": self.counts_nonpoem_2gram,
                "vocab": self.vocab,
                "centroid_poem_tfidf": self.centroid_poem_tfidf,
                "idf_arr": self.idf_arr,
                "train_settings": self.train_settings,
                "feat_names": self.feat_names,
            }, f)
        meta = {
            "path": str(path),
            "feat_names": self.feat_names,
            "n_features": len(self.feat_names),
            "train_settings": self.train_settings,
        }
        meta_path = path.with_suffix(".meta.json")
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "FrozenMetric":
        with Path(path).open("rb") as f:
            d = pickle.load(f)
        return cls(
            clf=d["clf"],
            scaler=d["scaler"],
            counts_poem=d["counts_poem"],
            counts_nonpoem=d["counts_nonpoem"],
            counts_poem_2gram=d["counts_poem_2gram"],
            counts_nonpoem_2gram=d["counts_nonpoem_2gram"],
            vocab=d["vocab"],
            centroid_poem_tfidf=d["centroid_poem_tfidf"],
            idf_arr=d["idf_arr"],
            train_settings=d["train_settings"],
        )


def build_and_freeze(seed: int = 42,
                     val_ratio: float = 0.2) -> FrozenMetric:
    """Re-run the Round-2 training pipeline and freeze the result.

    This is deterministic (given seed=42) and produces the same metric
    that Round 2 / Round 3 used.
    """
    samples = load_v2()
    train, val = train_val_split(samples, val_ratio=val_ratio, seed=seed)

    train_texts = [s.text for s in train]
    X_train = extract_batch(train_texts)
    y_train = np.asarray([s.label for s in train], dtype=np.int64)

    poems_in_train = [s for s in train if s.label == 1]
    nonpoems_in_train = [s for s in train if s.label == 0]
    counts_poem, total_poem = class_char_counts(poems_in_train, n=1)
    counts_nonpoem, total_nonpoem = class_char_counts(nonpoems_in_train, n=1)
    counts_poem_2gram, _ = class_char_counts(poems_in_train, n=2)
    counts_nonpoem_2gram, _ = class_char_counts(nonpoems_in_train, n=2)
    vocab = build_char_vocab(train_texts, min_df=2)
    centroid_poem_tfidf, idf_arr = build_tfidf_centroid(
        [s.text for s in poems_in_train], vocab)

    def base_feats(text: str) -> dict[str, float]:
        return {
            "bleu_to_poem": bleu_self(text, counts_poem, total_poem),
            "bleu_to_nonpoem": bleu_self(text, counts_nonpoem, total_nonpoem),
            "bigram_jacc_to_poem": bigram_jaccard_self(text, counts_poem_2gram),
            "bigram_jacc_to_nonpoem": bigram_jaccard_self(text, counts_nonpoem_2gram),
            "tfidf_cos_to_poem": char_tfidf_cosine_to_poetry_centroid(
                text, vocab, centroid_poem_tfidf, idf_arr),
        }

    base_train = np.asarray(
        [list(base_feats(t).values()) for t in train_texts], dtype=np.float64)
    X_train_full = np.concatenate([X_train, base_train], axis=1)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_full)

    clf = LogisticRegression(
        max_iter=3000, C=1.0, class_weight="balanced", random_state=seed)
    clf.fit(X_train_s, y_train)

    settings = {
        "seed": seed,
        "val_ratio": val_ratio,
        "n_train": len(train),
        "n_features": X_train_full.shape[1],
        "total_poem": total_poem,
        "total_nonpoem": total_nonpoem,
        "stage": "stage1_frozen",
        "based_on": "Round 2 / Round 3 dataset + classifier",
    }
    return FrozenMetric(
        clf=clf, scaler=scaler,
        counts_poem=counts_poem, counts_nonpoem=counts_nonpoem,
        counts_poem_2gram=counts_poem_2gram,
        counts_nonpoem_2gram=counts_nonpoem_2gram,
        vocab=vocab, centroid_poem_tfidf=centroid_poem_tfidf,
        idf_arr=idf_arr,
        train_settings=settings,
    )