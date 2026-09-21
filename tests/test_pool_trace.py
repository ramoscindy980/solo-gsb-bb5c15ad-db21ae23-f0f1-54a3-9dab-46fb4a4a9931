"""池化检测阳性溯源测试（不依赖 FastAPI，走 db + review + pool_trace 全链路）。

覆盖：
- 仅允许选择一条时效正常记录，读取其当前配方（两版取新版）叶子原料，
  同名叶子去重、非叶子（复合配料）不可引用；
- 「池名：原料列表：阳性/阴性」逐行解析：三段格式、状态仅接受阳性/阴性、
  注释空行、池名去空与重名、引用非配方叶子原料整批拒绝（逐行给出问题）；
- 阴性池排除其全部原料（即使同一原料也出现在阳性池中）；阳性池被阴性
  证据完全排除（含空原料列表）时无解并列出全部冲突池；
- 子集动态规划（最小命中集）：最少嫌疑原料数、并列解数与按原料名序列
  字典序取唯一最小解，对照暴力枚举；逐嫌疑原料列出其解释的阳性池；
- 无阳性池时空集合即解释全部；
- 缺失 / 非法 ID / 需复核 / 旧记录拒绝并说明；全程只读、不写库。
"""
import os
import tempfile
import unittest
from itertools import combinations

_DB_DIR = tempfile.mkdtemp(prefix="pool-trace-")
os.environ["DB_PATH"] = os.path.join(_DB_DIR, "trace.db")

from app import db  # noqa: E402
from app.pool_trace import (PoolTraceError, build_pool_trace_preview,  # noqa: E402
                            list_trace_records, parse_pools, solve_trace)
from app.recheck import build_recheck  # noqa: E402
from app.review import analyze  # noqa: E402

DICT_TEXT = """\
小麦粉：麸质谷物
乳清粉：乳
乳糖：乳
奶粉：乳
可可脂：
大豆卵磷脂：大豆
鸡蛋：蛋类
椰子粉：坚果
芝麻：芝麻
食盐：
"""
TEXT_MILK = "配料表：小麦粉，白砂糖，巧克力酱（可可脂，奶粉（乳清粉，乳糖），大豆卵磷脂），鸡蛋，食盐"
TEXT_B = "配料表：小麦粉，椰子粉，鸡蛋，芝麻，食盐"

LEAVES = ["小麦粉", "白砂糖", "可可脂", "乳清粉", "乳糖", "大豆卵磷脂", "鸡蛋", "食盐"]


def make_review(text_a, text_b=None, name="溯源测试", with_fingerprint=True):
    result, fp = analyze(text_a=text_a, text_b=text_b, label_a="现版配方",
                         label_b="新版配方", declared=[])
    return db.create_review(
        name=name, declared=[], text_a=text_a, text_b=text_b,
        label_a="现版配方", label_b="新版配方", result=result,
        dict_fingerprint=fp if with_fingerprint else None)


def brute_force(leaves, pools):
    """暴力枚举全部原料子集，返回 (最少原料数|None, 名称键, 方案数)。"""
    excluded = set()
    positives = []
    for p in pools:
        if p["positive"]:
            positives.append(set(p["ingredients"]) - excluded)
        else:
            excluded |= set(p["ingredients"])
    # 排除后重算阳性候选（阴性可能在后面行）
    positives = [set(p["ingredients"]) - excluded for p in pools if p["positive"]]
    if any(not s for s in positives):
        return None, None, 0
    universe = sorted({i for s in positives for i in s})
    best, best_key, ways = None, None, 0
    for r in range(len(universe) + 1):
        for combo in combinations(universe, r):
            chosen = set(combo)
            if not all(s & chosen for s in positives):
                continue
            key = tuple(sorted(combo))
            if best is None or r < best:
                best, best_key, ways = r, key, 1
            elif r == best:
                ways += 1
                if key < best_key:
                    best_key = key
    return best, best_key, ways


class LeavesTest(unittest.TestCase):
    """叶子原料提取：只取叶子、同名去重、复合配料不算叶子。"""

    def setUp(self):
        db.DB_PATH = os.path.join(_DB_DIR, f"leaves_{self.id().rsplit('.', 1)[-1]}.db")
        db.init_db()
        db.bulk_import(DICT_TEXT)

    def test_leaves_from_current_version(self):
        rid = make_review(TEXT_MILK, TEXT_B, name="两版")
        records = {r["id"]: r for r in list_trace_records()}
        rec = records[rid]
        self.assertEqual(rec["source_version"], "新版配方")
        # B 版叶子：小麦粉、椰子粉、鸡蛋、芝麻、食盐；巧克力酱 / 奶粉不出现
        self.assertEqual(rec["leaf_ingredients"],
                         ["小麦粉", "椰子粉", "鸡蛋", "芝麻", "食盐"])

    def test_leaf_names_are_tree_originals(self):
        rid = make_review(TEXT_MILK)
        leaves = {r["id"]: r for r in list_trace_records()}[rid]["leaf_ingredients"]
        self.assertEqual(leaves, LEAVES)


class ParseTest(unittest.TestCase):
    """池行解析：格式、状态、重名、非配方原料引用、注释。"""

    def _index(self, leaves=LEAVES):
        from app.analysis import normalize_name
        return {normalize_name(n): n for n in leaves}

    def test_basic_parse(self):
        pools = parse_pools(
            "# 注释\n\n阴性池：可可脂，食盐：阴性\n"
            "一混: 乳清粉 可可脂 : 阳性\n空阳：：阳性", self._index())
        self.assertEqual([p["name"] for p in pools], ["阴性池", "一混", "空阳"])
        self.assertFalse(pools[0]["positive"])
        self.assertTrue(pools[1]["positive"])
        self.assertEqual(pools[1]["ingredients"], ["乳清粉", "可可脂"])
        self.assertEqual(pools[2]["ingredients"], [])
    def test_format_status_duplicate_unknown(self):
        with self.assertRaises(PoolTraceError) as ctx:
            parse_pools(
                "只有名\n  ：乳清粉：阴性\nA：乳清粉：阳性\nA：乳糖：阴性\n"
                "B：火星原料：阳性\nC：乳清粉：+\nD：奶粉：阳性",
                self._index())
        types = [(p["line"], p["type"]) for p in ctx.exception.problems]
        self.assertEqual(types, [
            (1, "format"),       # 只有一段
            (2, "empty_name"),   # 空池名
            (4, "duplicate_name"),
            (5, "unknown_ingredient"),
            (6, "bad_status"),   # 不接受 +
            (7, "unknown_ingredient"),  # 奶粉是复合配料，不是叶子
        ])

    def test_english_colon_and_normalized_match(self):
        # 空白 / 大小写归一匹配，展示名取树中名称，列表按名排序
        pools = parse_pools("P:  乳清粉 , 可可脂:阳性", self._index())
        self.assertEqual(pools[0]["ingredients"], ["乳清粉", "可可脂"])

    def test_repeated_ingredient_in_pool_dedup(self):
        pools = parse_pools("P：乳清粉，乳清粉：阳性", self._index())
        self.assertEqual(pools[0]["ingredients"], ["乳清粉"])


class DPTest(unittest.TestCase):
    """命中集 DP 纯计算：最小集合、并列字典序（对照暴力枚举）、冲突、排除。"""

    def _pools(self, spec):
        return [{"name": n, "positive": pos, "ingredients": list(items)}
                for n, pos, items in spec]

    def test_basic_minimum_hit(self):
        pools = self._pools([
            ("N", False, ["可可脂", "食盐"]),
            ("P1", True, ["乳清粉", "可可脂"]),
            ("P2", True, ["大豆卵磷脂", "食盐"]),
            ("P3", True, ["鸡蛋", "白砂糖"]),
        ])
        r = solve_trace(LEAVES, pools)
        self.assertTrue(r["feasible"])
        self.assertEqual(r["minimum_size"], 3)
        self.assertEqual([s["ingredient"] for s in r["suspects"]],
                         ["乳清粉", "大豆卵磷脂", "白砂糖"])
        explained = {s["ingredient"]: s["explains_pools"]
                     for s in r["suspects"]}
        self.assertEqual(explained["乳清粉"], ["P1"])
        self.assertEqual(explained["大豆卵磷脂"], ["P2"])
        self.assertEqual(explained["白砂糖"], ["P3"])
        self.assertEqual(r["excluded_ingredients"], ["可可脂", "食盐"])

    def test_one_covers_all(self):
        pools = self._pools([
            ("N", False, ["食盐"]),
            ("P1", True, ["乳清粉", "鸡蛋"]),
            ("P2", True, ["乳清粉", "白砂糖"]),
        ])
        r = solve_trace(LEAVES, pools)
        self.assertTrue(r["feasible"])
        self.assertEqual(r["minimum_size"], 1)
        self.assertEqual(r["optimal_count"], 1)
        self.assertEqual([s["ingredient"] for s in r["suspects"]], ["乳清粉"])
        self.assertEqual(sorted(r["suspects"][0]["explains_pools"]),
                         ["P1", "P2"])

    def test_tie_break_and_count(self):
        # P3={鸡蛋,白砂糖}，P1/P2 可由乳清粉一个命中：
        # 两个最小集（乳清粉,鸡蛋）与（乳清粉,白砂糖）并列，
        # 字典序取原料名序列更小者：白(U+767D) < 鸡(U+9E21) → 白砂糖
        pools = self._pools([
            ("N", False, ["可可脂", "食盐"]),
            ("P1", True, ["乳清粉", "可可脂"]),
            ("P2", True, ["大豆卵磷脂", "乳清粉", "食盐"]),
            ("P3", True, ["鸡蛋", "白砂糖"]),
        ])
        r = solve_trace(LEAVES, pools)
        self.assertTrue(r["feasible"])
        self.assertEqual(r["minimum_size"], 2)
        self.assertEqual(r["optimal_count"], 2)
        self.assertEqual([s["ingredient"] for s in r["suspects"]],
                         ["乳清粉", "白砂糖"])

    def test_conflict_positive_pool_excluded(self):
        # 阳性池全部成员都在阴性池中（原料同时出现在阴 / 阳池 → 被排除）
        pools = self._pools([
            ("N1", False, ["可可脂", "食盐"]),
            ("P1", True, ["食盐"]),
            ("P2", True, ["可可脂", "乳清粉"]),
        ])
        r = solve_trace(LEAVES, pools)
        self.assertFalse(r["feasible"])
        self.assertEqual([c["name"] for c in r["conflict_pools"]], ["P1"])
        self.assertEqual(r["conflict_pools"][0]["excluded_ingredients"], ["食盐"])
        self.assertIsNone(r["minimum_size"])

    def test_empty_positive_pool_conflicts(self):
        r = solve_trace(LEAVES, self._pools([("空阳", True, [])]))
        self.assertFalse(r["feasible"])
        self.assertEqual(r["conflict_pools"][0]["name"], "空阳")
        self.assertEqual(r["conflict_pools"][0]["ingredients"], [])

    def test_only_negative_pools_empty_solution(self):
        r = solve_trace(LEAVES, self._pools([
            ("N1", False, ["食盐"]), ("N2", False, ["可可脂"])]))
        self.assertTrue(r["feasible"])
        self.assertEqual(r["minimum_size"], 0)
        self.assertEqual(r["optimal_count"], 1)
        self.assertEqual(r["suspects"], [])

    def test_no_pools_at_all(self):
        r = solve_trace(LEAVES, [])
        self.assertTrue(r["feasible"])
        self.assertEqual(r["minimum_size"], 0)
        self.assertEqual(r["suspects"], [])

    def test_duplicate_positive_pools_count_once_each(self):
        # 两个内容完全相同的阳性池：一个嫌疑原料同时命中两者
        pools = self._pools([
            ("P1", True, ["乳清粉", "鸡蛋"]),
            ("P2", True, ["乳清粉", "鸡蛋"]),
        ])
        r = solve_trace(LEAVES, pools)
        self.assertEqual(r["minimum_size"], 1)
        self.assertEqual([s["ingredient"] for s in r["suspects"]], ["乳清粉"])
        self.assertEqual(sorted(r["suspects"][0]["explains_pools"]),
                         ["P1", "P2"])

    def test_matches_brute_force(self):
        import random
        random.seed(20260921)
        universe = [f"原料{c}" for c in "ABCDEF"]
        for trial in range(400):
            spec = []
            for i in range(random.randint(1, 6)):
                items = random.sample(universe, random.randint(0, 4))
                spec.append((f"P{i}", random.random() < 0.6, items))
            # 池名由 parse 阶段保证唯一；纯计算直接造池
            pools = self._pools(spec)
            r = solve_trace(universe, pools)
            bc, bkey, bways = brute_force(universe, pools)
            if bc is None:
                self.assertFalse(r["feasible"])
                # 每个冲突阳性池的候选原料确实为空（全被阴性排除）
                excluded = set()
                for p in pools:
                    if not p["positive"]:
                        excluded |= set(p["ingredients"])
                for c in r["conflict_pools"]:
                    self.assertFalse(set(c["ingredients"]) - excluded)
            else:
                self.assertTrue(r["feasible"])
                self.assertEqual(r["minimum_size"], bc)
                self.assertEqual(r["optimal_count"], bways)
                key = tuple(s["ingredient"] for s in r["suspects"])
                self.assertEqual(key, bkey)
                # 每个阳性池至少被一个所选嫌疑命中
                chosen = set(key)
                for p in pools:
                    if p["positive"]:
                        self.assertTrue(chosen & set(p["ingredients"]))
                # 逐嫌疑解释的阳性池清单与实际命中一致
                for s in r["suspects"]:
                    hit = [p["name"] for p in pools if p["positive"]
                           and s["ingredient"] in p["ingredients"]]
                    self.assertEqual(s["explains_pools"], hit)


class PreviewFlowTest(unittest.TestCase):
    """全链路：记录资格、两版取 B、冲突、需复核 / 缺失拒绝、只读。"""

    def setUp(self):
        db.DB_PATH = os.path.join(
            _DB_DIR, f"trace_{self.id().rsplit('.', 1)[-1]}.db")
        db.init_db()
        db.bulk_import(DICT_TEXT)

    def _fp(self, rid):
        with db._connect() as conn:
            return conn.execute(
                "SELECT dict_fingerprint FROM reviews WHERE id=?",
                (rid,)).fetchone()[0]

    def test_leaves_from_two_version_record(self):
        rid = make_review(TEXT_MILK, TEXT_B, name="两版")
        r = build_pool_trace_preview(rid, "P：椰子粉：阳性")
        self.assertEqual(r["review"]["leaf_ingredients"],
                         ["小麦粉", "椰子粉", "鸡蛋", "芝麻", "食盐"])
        self.assertTrue(r["feasible"])
        self.assertEqual(r["suspects"][0]["ingredient"], "椰子粉")

    def test_feasible_trace(self):
        rid = make_review(TEXT_MILK)
        text = ("N：可可脂，食盐：阴性\n"
                "一混：乳清粉，可可脂：阳性\n"
                "二混：大豆卵磷脂，食盐：阳性\n"
                "三混：鸡蛋，白砂糖：阳性")
        r = build_pool_trace_preview(rid, text)
        self.assertTrue(r["feasible"])
        self.assertEqual(r["minimum_size"], 3)
        self.assertEqual(r["optimal_count"], 2)
        self.assertEqual([s["ingredient"] for s in r["suspects"]],
                         ["乳清粉", "大豆卵磷脂", "白砂糖"])
        self.assertEqual(r["negative_pools"], ["N"])

    def test_conflict_trace(self):
        rid = make_review(TEXT_MILK)
        r = build_pool_trace_preview(
            rid, "N：可可脂：阴性\nP：可可脂：阳性")
        self.assertFalse(r["feasible"])
        self.assertEqual([c["name"] for c in r["conflict_pools"]], ["P"])

    def test_batch_rejected_with_lines(self):
        rid = make_review(TEXT_MILK)
        with self.assertRaises(PoolTraceError) as ctx:
            build_pool_trace_preview(
                rid,
                "P：火星原料：阳性\nP：乳清粉：阳性\nQ：乳清粉：可疑\n坏")
        self.assertEqual([(p["line"], p["type"]) for p in ctx.exception.problems],
                         [(1, "unknown_ingredient"), (2, "duplicate_name"),
                          (3, "bad_status"), (4, "format")])

    def test_empty_pools_rejected(self):
        rid = make_review(TEXT_MILK)
        with self.assertRaises(PoolTraceError) as ctx:
            build_pool_trace_preview(rid, "# 只有注释\n")
        self.assertEqual(ctx.exception.problems[0]["type"], "empty")

    def test_missing_and_invalid_record_blocked(self):
        with self.assertRaises(PoolTraceError) as ctx:
            build_pool_trace_preview(999999, "P：乳清粉：阳性")
        self.assertEqual(ctx.exception.problems[0]["reason"], "missing")
        with self.assertRaises(PoolTraceError) as ctx:
            build_pool_trace_preview("x", "P：乳清粉：阳性")
        self.assertEqual(ctx.exception.problems[0]["reason"], "invalid")

    def test_stale_record_blocked_until_rechecked(self):
        rid = make_review(TEXT_MILK)
        db.upsert_entry("乳清粉", ["乳", "花生"])  # 映射变更 → 需复核
        self.assertTrue(db.get_review(rid)["needs_recheck"])
        with self.assertRaises(PoolTraceError) as ctx:
            build_pool_trace_preview(rid, "P：乳清粉：阳性")
        [p] = ctx.exception.problems
        self.assertEqual(p["reason"], "stale")
        self.assertIn("需复核", p["message"])
        records = {r["id"]: r for r in list_trace_records()}
        self.assertFalse(records[rid]["eligible"])
        outcome = build_recheck(db.get_review(rid))
        db.update_review_result(rid, outcome["new_result"],
                                outcome["fingerprint"])
        self.assertFalse(db.get_review(rid)["needs_recheck"])
        r = build_pool_trace_preview(rid, "P：乳清粉：阳性")
        self.assertTrue(r["feasible"])

    def test_legacy_record_blocked(self):
        rid = make_review(TEXT_MILK, with_fingerprint=False)
        with self.assertRaises(PoolTraceError) as ctx:
            build_pool_trace_preview(rid, "P：乳清粉：阳性")
        self.assertEqual(ctx.exception.problems[0]["reason"], "legacy")

    def test_deleted_record_blocked(self):
        rid = make_review(TEXT_MILK)
        db.delete_review(rid)
        with self.assertRaises(PoolTraceError) as ctx:
            build_pool_trace_preview(rid, "P：乳清粉：阳性")
        self.assertEqual(ctx.exception.problems[0]["reason"], "missing")

    def test_readonly(self):
        rid = make_review(TEXT_MILK)
        before = db.get_review(rid)
        count = len(db.list_reviews())
        text = ("N：可可脂，食盐：阴性\n"
                "一混：乳清粉，可可脂：阳性\n"
                "二混：大豆卵磷脂，食盐：阳性\n"
                "三混：鸡蛋，白砂糖：阳性")
        build_pool_trace_preview(rid, text)
        build_pool_trace_preview(rid, "N：乳清粉：阴性\nP：乳清粉：阳性")  # 无解也不写库
        self.assertEqual(db.get_review(rid), before)
        self.assertEqual(self._fp(rid), before["dict_fingerprint"])
        self.assertEqual(len(db.list_reviews()), count)


if __name__ == "__main__":
    unittest.main()
