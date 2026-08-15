# Check-Agent · 审计规则（不可修改）

> 本文件是 Check-Agent 的唯一规则源。Agent 不得通过任何方式修改。

## 不可修改规则清单

1. 数据划分（train / val / test）
2. 测试集访问权限
3. 人工评价标签
4. 核心评价协议
5. 实验目标
6. Plan / 提示词 / Skills

## 越界行为分类

| 类型       | 例子                                                      | 处理                  |
| ---------- | --------------------------------------------------------- | --------------------- |
| DATA_OOB   | 访问未声明的数据路径或读取 test 划分                       | 标记 INVALID，写 rules_memory |
| RULE_BYPASS| 通过修改 Plan / 提示词 / Skills 绕过约束                    | 标记 INVALID，写 rules_memory |
| PROTOCOL_CHANGE | 修改评测协议或人类标签                                 | 标记 INVALID，写 rules_memory |
| WRITE_OOB  | 在 `E:\生成诗歌\` 下任何位置写入                            | 标记 INVALID，写 rules_memory |
| UNAUTH_API | 使用未声明的模型 / API / 外部数据                          | 标记 INVALID，写 rules_memory |

## INVALID 标记规范

见 `invalid_marker.md`。

## 审计周期

- 每轮实验**前后**必做边界检查（pre_round / post_round）
- 每**3 轮**一次完整审计（full_audit）
- 见 `audit_cycle.md`