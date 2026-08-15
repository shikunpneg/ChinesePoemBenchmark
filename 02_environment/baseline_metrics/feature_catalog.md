# 基础特征库（baseline feature catalog）

> Agent 可使用的**唯一特征来源**。任何不在本目录的特征，由 Check-Agent 标记为越界。

## 类别 1：词面与统计特征

| 特征 ID     | 名称              | 描述                              |
| ----------- | ----------------- | --------------------------------- |
| F-stat-len  | length            | 字符 / 词数                        |
| F-stat-ttr  | type-token ratio  | 类符/形符比                        |
| F-stat-stop | stopword ratio    | 停用词比例                        |
| F-stat-punc | punctuation ratio | 标点比例                          |

## 类别 2：形式与韵律特征（中文专属）

| 特征 ID       | 名称             | 描述                                |
| ------------- | ---------------- | ----------------------------------- |
| F-form-line   | line_count       | 行数                                |
| F-form-couplet| couplet_consist  | 对仗一致性（粗略）                   |
| F-form-tone   | tone_pattern     | 平仄模式（粗略）                     |
| F-form-rhyme  | rhyme_score      | 押韵得分                            |

## 类别 3：传统 NLP 评价（平凡解）

| 特征 ID     | 名称    | 描述                       |
| ----------- | ------- | -------------------------- |
| F-nlp-bleu  | BLEU    | n-gram 重叠                |
| F-nlp-rouge | ROUGE   | 参考文本相似度              |
| F-nlp-edit  | edit distance | 词面编辑距离           |

## 类别 4：词汇语义 / 主题

| 特征 ID        | 名称                | 描述                                |
| -------------- | ------------------- | ----------------------------------- |
| F-sem-tfidf    | tfidf_diversity     | TF-IDF 类符多样性                   |
| F-sem-topic    | topic_dist          | 主题分布（可用 LDA / 主题词表）     |
| F-sem-embedding | sentence_emb_dist  | 句向量与语料均值的余弦距离           |

## 类别 5：LLM-as-a-Judge（受限使用）

| 特征 ID        | 名称              | 描述                                |
| -------------- | ----------------- | ----------------------------------- |
| F-llm-judge    | llm_score         | DeepSeek-V4-Flash 评分（受限规模）   |

> LLM-as-a-Judge 在本研究中**仅作为对比基线**，**不**作为优化目标。Agent 不得以 LLM 评分作为人类一致性的代理。

## 类别 6：意象 / 情感（粗略）

| 特征 ID       | 名称              | 描述                              |
| ------------- | ----------------- | --------------------------------- |
| F-img-density | image_density     | 意象词密度（基于粗略词表）         |
| F-emo-dist    | emotion_dist      | 情感词分布                         |
| F-figurative  | figurative_score  | 修辞手法（比喻 / 拟人 / 通感）粗略 |

## 落地原则

- 每个特征必须有：`id` / `description` / `inputs` / `outputs` / `range` / `license_or_origin`
- Agent 只能**调用**这些特征的实现，不得自行修改或新增。
- 下一轮交付：补全每个特征的 Python 实现 + 单元测试 + 输入输出契约。