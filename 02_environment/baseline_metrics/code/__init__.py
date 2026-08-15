"""Baseline metric implementations for stage-1 indicator combination search.

Each metric returns a value in [0, 1] where higher = more "poem-like".
The classifier (logistic regression) is trained downstream; this module
only computes features.
"""

from .features import (
    FEATURE_NAMES,
    extract_all_features,
    extract_batch,
    feat_form,
    feat_language,
    feat_logic_jump,
    feat_music_simple,
    feat_purity,
    feat_style,
    feat_structure,
    text_reliability,
)

from .meter import analyze_meter, meter_to_features
from .structure import structure_v2_features
from .imagery_ner import extract_entities, imagery_features, imagery_logic_jump
from .phonetics import phonetic_features

from .baselines import (
    bigram_jaccard_self,
    bleu_self,
    build_char_vocab,
    build_tfidf_centroid,
    char_counts,
    char_tfidf_cosine_to_poetry_centroid,
    rouge_l_self,
)

from .llm_judge import (
    LLMJudgeAPI,
    LLMJudgeBase,
    LLMJudgeResult,
    LLMJudgeStub,
    default_judge,
)

from .frozen_metric import FrozenMetric, FrozenPrediction, build_and_freeze

__all__ = [
    "FEATURE_NAMES",
    "extract_all_features",
    "extract_batch",
    "feat_form",
    "feat_language",
    "feat_logic_jump",
    "feat_music_simple",
    "feat_purity",
    "feat_style",
    "feat_structure",
    "text_reliability",
    "analyze_meter",
    "meter_to_features",
    "structure_v2_features",
    "extract_entities",
    "imagery_features",
    "imagery_logic_jump",
    "phonetic_features",
    "bigram_jaccard_self",
    "bleu_self",
    "build_char_vocab",
    "build_tfidf_centroid",
    "char_counts",
    "char_tfidf_cosine_to_poetry_centroid",
    "rouge_l_self",
    "LLMJudgeAPI",
    "LLMJudgeBase",
    "LLMJudgeResult",
    "LLMJudgeStub",
    "default_judge",
    "FrozenMetric",
    "FrozenPrediction",
    "build_and_freeze",
]