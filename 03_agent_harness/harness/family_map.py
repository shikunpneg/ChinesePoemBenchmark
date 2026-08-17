"""Feature family mapping — 13 族 × 64 维特征.

This is the canonical source for family-level feature selection in the
automated Harness. Each family maps to a subset of FEATURE_NAMES, so
the search space is 2^13 = 8192 combinations (manageable for
greedy/random exploration).

To add a new feature, edit `FEATURE_FAMILIES` AND `code/features.py`.
"""
from __future__ import annotations

# 13 族 × 64 维特征映射
# 与 code.features.FEATURE_NAMES 必须严格一致（顺序与命名）
FEATURE_FAMILIES: dict[str, list[str]] = {
    # 1) 近体诗格律（P1，v7 新增）
    "meter": [
        "meter_form_score",
        "meter_line_pattern_ok",
        "meter_dui_ok",
        "meter_nian_ok",
        "meter_rhyme_agreement",
        "meter_parallelism",
        "meter_is_metrical",
    ],
    # 2) 段落统计（para_theme 族）
    "para": [
        "para_para_count",
        "para_para_len_mean",
        "para_para_len_cv",
        "para_para_var",
    ],
    # 3) 主题分析（theme5）
    "theme": [
        "theme_theme_jump_mean",
        "theme_theme_jump_cv",
        "theme_theme_coherence",
        "theme_theme_cluster_ratio",
        "theme_opening_closure",
    ],
    # 4) 主题分析 8 维（theme8，v8 修稀疏性）
    "theme8": [
        "theme8_theme_jump_mean",
        "theme8_theme_jump_cv",
        "theme8_theme_coherence",
        "theme8_theme_cluster_ratio",
        "theme8_opening_closure",
        "theme8_unit_count_norm",
    ],
    # 5) 视觉结构（struct）
    "struct": [
        "struct_n_lines",
        "struct_line_ending_punct",
        "struct_short_line_ratio",
    ],
    # 6) NER + 意象场顺序（ner_img，v8 用 bge）
    "ner_img": [
        "ner_entity_density",
        "ner_field_diversity",
        "ner_field_sequence_len",
        "img_ent_adj_sim_mean",
        "img_ent_adj_sim_cv",
        "img_field_switch_rate",
        "img_field_return",
        "img_rupture_bridge",
        "img_logic_jump_score",
    ],
    # 7) bge 语义向量（sem）
    "sem": [
        "sem_adj_line_sim_mean",
        "sem_adj_line_sim_cv",
        "sem_first_last_sim",
        "sem_bridge_rate",
        "sem_wholeness",
        "sem_dispersion",
    ],
    # 8) 逻辑跳跃粗代理（jump）
    "jump": [
        "jump_connector_density",
        "jump_char_per_line",
        "jump_line_density_var",
    ],
    # 9) 词汇诗性（lang）
    "lang": [
        "lang_imagery_density",
        "lang_classical_marker_density",
        "lang_prose_particle_density",
        "lang_line_break_existence",
    ],
    # 10) 文本纯净度（purity，v3+）
    "purity": [
        "purity_han_ratio",
        "purity_no_english",
        "purity_no_digit",
        "purity_line_cleanliness",
    ],
    # 11) 语域信号（style，v5+）
    "style": [
        "style_news_word_density",
        "style_news_phrase_density",
        "style_forum_filler_density",
        "style_avg_para_len",
    ],
    # 12) 平仄简化（music，历史保留）
    "music": [
        "music_pattern_regularity",
        "music_ping_ze_balance",
        "music_final_char_ping_ratio",
    ],
    # 13) 真实声学信号（phon，v7+）
    "phon": [
        "phon_tone_smoothness",
        "phon_tone_cv",
        "phon_resonance_var",
        "phon_rhyme_distance",
        "phon_rhyme_repeat",
        "phon_tone_balance",
    ],
}


# 验证：所有 13 族特征数之和 == 64
def _verify() -> None:
    total = sum(len(v) for v in FEATURE_FAMILIES.values())
    assert total == 64, (
        f"族特征总数 {total} ≠ 64，请同时更新 family_map.py 和 features.py"
    )
    all_names = []
    for fams in FEATURE_FAMILIES.values():
        all_names.extend(fams)
    assert len(set(all_names)) == 64, "族之间有重复特征"


_verify()


ALL_FAMILIES: list[str] = list(FEATURE_FAMILIES.keys())
N_FAMILIES: int = len(ALL_FAMILIES)


def build_family_mask(active_families: list[str]) -> list[bool]:
    """构造 64 维 mask：active_families 内的特征 True，其他 False。

    Args:
        active_families: 启用的族名列表（必须都在 FEATURE_FAMILIES 中）

    Returns:
        64 长度的 bool list，与 FEATURE_NAMES 顺序对应
    """
    unknown = set(active_families) - set(FEATURE_FAMILIES)
    if unknown:
        raise ValueError(f"未知族: {unknown}")
    mask: list[bool] = []
    for fam in FEATURE_FAMILIES:
        for _ in FEATURE_FAMILIES[fam]:
            mask.append(fam in active_families)
    return mask


def get_all_family_features() -> list[str]:
    """返回所有 64 维特征名（按 FEATURE_FAMILIES 顺序）。"""
    feats: list[str] = []
    for fam in FEATURE_FAMILIES:
        feats.extend(FEATURE_FAMILIES[fam])
    return feats
