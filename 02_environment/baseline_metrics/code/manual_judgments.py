"""Manual judgment analysis of the 215 disagreements.

I'll classify each reviewed sample as:
  - 'indicator_correct': indicator matches truth (text), humans wrong (voted by metadata)
  - 'humans_correct':   humans match truth (text), indicator wrong
  - 'both_wrong':       both wrong (rare)
  - 'ambiguous':        cannot tell from text alone

Stratification by (hum_label, ind_label, genre, source_type).
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

DIS_PATH = Path(r"E:\ai4s\poetry-poetricity\05_experiments\stage2_hard_samples\round_004\disagreements_to_review.json")

# My manual judgments for the reviewed samples.
# key = sample_id (int), value = "indicator_correct" | "humans_correct" | "both_wrong" | "ambiguous"
JUDGMENTS = {
    # hum=1, ind=0 (humans say poem, indicator says no) - 28 samples
    # Most had scrambled metadata: title=poem, text=news
    # Annotators voted based on title (误以为 is poem); indicator judged by text (correctly NOT poem)
    594: "indicator_correct",   # title=无题, text=financial news
    770: "indicator_correct",   # title=运动套装, text=西川诗歌课 news
    689: "indicator_correct",   # title=无题, text=冰箱 social
    637: "indicator_correct",   # title=无题, text=佳佳 social
    247: "indicator_correct",   # title=徐志摩, text=4句 (looks like classical)
    617: "indicator_correct",   # title=无题, text=地铁 social
    732: "indicator_correct",   # title=拯救卤味, text=AI学院 social
    707: "indicator_correct",   # title=机器人, text=评论 social
    638: "indicator_correct",   # title=某国, text=城市 social
    413: "indicator_correct",   # title=出塞王昌龄, text=娱乐 news (SCRAMBLED!)
    597: "indicator_correct",   # title=交易, text=韩国果树
    417: "indicator_correct",   # title=长信怨王昌龄, text=陆毅 news (SCRAMBLED)
    792: "indicator_correct",   # title=无题, text=评论海子
    473: "indicator_correct",   # title=山寺杜牧, text=全球经济 (SCRAMBLED)
    666: "indicator_correct",   # title=忆江南, text=海子诗
    581: "indicator_correct",   # title=地狱占星, text=信用卡
    460: "indicator_correct",   # title=绿萝杜牧, text=NBA (SCRAMBLED)
    661: "indicator_correct",   # title=忆江南, text=忆江南诗 (4-line poem)
    471: "indicator_correct",   # title=杜牧, text=欧洲央行 (SCRAMBLED)
    401: "indicator_correct",   # title=刘长卿, text=都市离婚 (SCRAMBLED)
    592: "indicator_correct",   # title=也许, text=QDII
    499: "indicator_correct",   # title=杜牧, text=小康社会 (SCRAMBLED)
    407: "indicator_correct",   # title=望岳杜甫, text=昆汀电影 (SCRAMBLED)
    420: "indicator_correct",   # title=春思李白, text=冯女郎 (SCRAMBLED)
    579: "indicator_correct",   # title=无题, text=斗地主
    408: "indicator_correct",   # title=别房太尉墓杜甫, text=霍华德电影 (SCRAMBLED)
    53:  "ambiguous",          # title=友谊顾城, text=顾城真诗 — indicator prob 0.42, pred 0
    418: "indicator_correct",   # title=岑参, text=黎明电影 (SCRAMBLED)

    # hum=0, ind=1, poem+classic (3 samples)
    # All 3 had DB labeled as poem but actual text was news → indicator was WRONG
    450: "humans_correct",   # title=鸦杜牧, text=NBA news (SCRAMBLED)
    497: "humans_correct",   # title=自贻杜牧, text=陈水扁 (SCRAMBLED)
    431: "humans_correct",   # title=即事黄州作杜牧, text=监外执行 (SCRAMBLED)

    # hum=0, ind=1, poem+modern (15 sampled)
    # Most had real classical poetry text, label scrambled
    # Indicator correctly identifies poems
    91:  "indicator_correct",   # 海子 real poem
    118: "indicator_correct",   # 海子 real poem
    127: "indicator_correct",   # 张枣 real poem
    178: "indicator_correct",   # 张枣 徐志摩 real poem (long)
    190: "indicator_correct",   # 徐志摩 real poem
    203: "indicator_correct",   # 徐志摩 古典五言 real poem (sourrce_type=modern wrong)
    212: "indicator_correct",   # 徐志摩 古典七言 real poem
    215: "indicator_correct",   # 徐志摩 古典五言 real poem
    216: "indicator_correct",   # 徐志摩 古典五言 real poem
    218: "indicator_correct",   # 徐志摩 古典五言 real poem
    220: "indicator_correct",   # 徐志摩 古典五言 real poem
    227: "indicator_correct",   # 徐志摩 古典五言 real poem
    232: "indicator_correct",   # 徐志摩 古典五言 real poem
    237: "indicator_correct",   # 徐志摩 古典五言 real poem
    242: "indicator_correct",   # 徐志摩 古典五言 real poem

    # hum=0, ind=1, nonpoem+news (15 sampled)
    # All 15 are actually classical poems (DB mislabeled)
    # Indicator correctly identifies poems
    804: "indicator_correct",  # title=性格分析, text=银地无尘金菊开 (古诗)
    810: "indicator_correct",  # title=无题, text=不是难提挈
    819: "indicator_correct",  # title=追踪被摄, text=外事因慵废
    859: "indicator_correct",  # title=2晋级, text=元精回复
    865: "indicator_correct",  # title=雅虎, text=别馆分周国
    871: "indicator_correct",  # title=无题, text=圣主何曾识仲都
    917: "indicator_correct",  # title=创业园区, text=在富莫骄奢
    920: "indicator_correct",  # title=无题, text=草堂琴画已判烧
    930: "indicator_correct",  # title=操心命, text=念子为儒道未亨
    941: "indicator_correct",  # title=无题, text=独凭朱槛亦凌晨
    943: "indicator_correct",  # title=搜集, text=汉室欢娱盛
    969: "indicator_correct",  # title=无题, text=城隅有乐游 (long)
    985: "indicator_correct",  # title=一点思考, text=清浅萦纡一水间
    988: "indicator_correct",  # title=无题, text=家家生计只琴书
    993: "indicator_correct",  # title=无题, text=花时曾省杜陵游

    # hum=0, ind=1, nonpoem+social (17 sampled)
    # All 17 are genuine social/news/finance text — humans correct, indicator wrong
    553: "humans_correct",   # 车模内衣
    542: "humans_correct",   # 泡泡裙 fashion
    524: "humans_correct",   # 七夕情人节
    539: "humans_correct",   # 邓丽君咖啡店
    551: "humans_correct",   # Topshop prices
    510: "humans_correct",   # 爱国者A5 specs
    567: "humans_correct",   # 移动短信
    550: "humans_correct",   # 蝴蝶结 fashion
    502: "humans_correct",   # 红毛猩猩医院
    504: "humans_correct",   # 尼康S640 specs
    562: "humans_correct",   # 游戏任务
    536: "humans_correct",   # 哈里马球
    535: "humans_correct",   # Bowdoin college
    547: "humans_correct",   # 情人节粉色
    566: "humans_correct",   # 劳拉造型
    546: "humans_correct",   # 杜鹃模特
    596: "humans_correct",   # 基金经理
}

items = json.loads(DIS_PATH.read_text(encoding="utf-8"))

# Tally
by_direction = defaultdict(list)
for d in items:
    if d["majority_label"] == 0 and d["indicator_pred"] == 1:
        direction = "hum=0_ind=1"
    elif d["majority_label"] == 1 and d["indicator_pred"] == 0:
        direction = "hum=1_ind=0"
    else:
        direction = "other"
    by_direction[direction].append(d)

# Count my judgments per direction × category
print("=== TALLY OF MANUAL JUDGMENTS (reviewed subset) ===\n")

tallies = Counter()
by_cat = defaultdict(Counter)
for sid, j in JUDGMENTS.items():
    tallies[j] += 1
    item = next((d for d in items if d["sample_id"] == sid), None)
    if item:
        cat = (item["genre"], item["source_type"])
        d = ("hum=1,ind=0" if item["majority_label"] == 1 and item["indicator_pred"] == 0
             else "hum=0,ind=1")
        by_cat[(d, cat)][j] += 1

print(f"Reviewed: {len(JUDGMENTS)} / 215 disagreements")
print()
print("=== by (direction, category) ===")
for (d, cat), c in sorted(by_cat.items()):
    print(f"  {d}  genre={cat[0]} source={cat[1]:8s}  n={sum(c.values())}  -> {dict(c)}")

print()
print("=== overall ===")
for j, n in sorted(tallies.items()):
    print(f"  {j}: {n} ({n/len(JUDGMENTS)*100:.1f}%)")

# Project findings to full population
print()
print("=== PROJECTION TO FULL 215 (assume reviewed sample is representative) ===")
direction_breakdown = Counter()
for sid, j in JUDGMENTS.items():
    item = next((d for d in items if d["sample_id"] == sid), None)
    if item:
        d = ("hum=1,ind=0" if item["majority_label"] == 1 and item["indicator_pred"] == 0
             else "hum=0,ind=1")
        direction_breakdown[d] += 1

# total counts per direction
total_per_direction = Counter(d["direction"] for d in by_direction["hum=1_ind=0"]) + Counter()
print(f"  hum=1,ind=0: {len(by_direction['hum=1_ind=0'])} total, reviewed {direction_breakdown.get('hum=1,ind=0',0)}")

# By my judgment, scale to population
# If reviewed 28 hum=1,ind=0 and 27 were indicator_correct, then in full 28, expect ~27 indicator_correct
print()
print("=== EXPECTED OUTCOMES IF REVIEWED SAMPLE IS REPRESENTATIVE ===")
n_reviewed = sum(tallies.values())
for j, n in tallies.items():
    pct = n / n_reviewed
    print(f"  {j}: {pct*100:.1f}% of disagreements")
    print(f"    -> extrapolated to 215 total: ~{int(215 * pct)} samples would be {j}")