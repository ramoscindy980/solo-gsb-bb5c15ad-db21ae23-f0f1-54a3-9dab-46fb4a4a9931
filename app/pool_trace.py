"""池化检测阳性溯源（临时预览，不改写审查记录）。

从历史审查记录中**仅选择一条时效正常**的记录，读取其**当前配方**（两版
记录取标签核对对象，即新版配方；单版取现版）的**叶子原料**（配料树中没有
子配料的节点；同名叶子多次出现只保留一个展示名，匹配按归一名去空白大小写）。

用户逐行录入池化检测结果，每行格式为::

    池名：原料列表：阳性/阴性

- 冒号中英文均可，必须恰好给出三段；原料之间用逗号 / 顿号 / 分号 /
  空白分隔（与字典批量导入口径一致）；状态只接受 ``阳性`` / ``阴性``
  （英文 ``positive`` / ``negative``、``+`` / ``-`` 一律视为非法）；
- 原料列表引用的每种原料都必须是所选记录当前配方的**叶子原料**，引用
  非配方叶子原料的池行整批拒绝；
- 池名不能为空、同一批内池名不得重复；``#`` 开头的行与空行忽略；
- 原料列表允许为空：阴性空池不排除任何原料；阳性空池不含任何嫌疑原料，
  必然无法被解释，作为冲突池返回**无解**。

溯源规则（集合覆盖 / 最小命中集）：

- **阴性池**排除其全部原料：出现在任一阴性池中的叶子原料一律不可能是
  阳性来源（即使它同时出现在阳性池中）；
- **阳性池**的检测阳性必须由池内至少一个嫌疑原料解释，故每个阳性池至少
  要包含一个未被阴性证据排除的原料，否则直接**无解**并给出全部冲突池；
- 后端在「原料 × 阳性池」上做**子集动态规划**（命中集 DP，状态数 =
  2^|阳性池|），求能解释（命中）**全部阳性池**的**最少嫌疑原料集合**：
  统计达到最小集合大小的全部并列解数量，并按**原料名升序序列字典序**
  重建唯一最小者展示；同时列出每个嫌疑原料各自解释（命中）的阳性池。

记录缺失（不存在 / 已删除）或处于**需复核**状态（旧记录无指纹、相关原料
映射已变更）时拒绝溯源并说明原因。整个模块只读：不写入任何数据，也不
改写任何审查记录。
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from . import db
from .analysis import normalize_name
from .changeover import blocked_message

_SPLIT_ITEMS_RE = re.compile(r"[,，、;；\s]+")
_SPLIT_COLON_RE = re.compile(r"[:：]")

# 状态词 → 是否阳性；只接受中文「阳性 / 阴性」
_STATUS = {"阳性": True, "阴性": False}


class PoolTraceError(ValueError):
    """阳性溯源请求无法完成（映射为 HTTP 400）。

    problems 逐条列出原因：记录资格问题（缺失 / 需复核 / ID 无效）或池
    行解析问题（行号、类型、原文），供页面定位。
    """

    def __init__(self, message: str, problems: Optional[List[dict]] = None):
        super().__init__(message)
        self.problems = problems or []


def list_trace_records() -> List[dict]:
    """候选记录清单：每条历史记录当前配方的叶子原料与选择资格。"""
    records = []
    for summary in db.list_reviews():
        review = db.get_review(summary["id"])
        needs = bool(review.get("needs_recheck"))
        reason = review.get("recheck_reason")
        records.append({
            "id": review["id"],
            "name": review["name"],
            "created_at": review["created_at"],
            "has_two_versions": bool(review.get("text_b")),
            "source_version": _source_version(review),
            "leaf_ingredients": _leaf_ingredients(_current_tree(review)),
            "eligible": not needs,
            "blocked_reason": blocked_message(reason) if needs else None,
        })
    records.sort(key=lambda r: r["id"])
    return records


def _record_problem(review_id, reason: str, message: str,
                    name: Optional[str] = None) -> dict:
    return {"id": review_id, "name": name, "reason": reason, "message": message}


def _current_version(review: dict) -> dict:
    """记录当前配方（标签核对对象：有新版取新版，否则取现版）的版本结果。"""
    result = review.get("result") or {}
    return result.get("version_b") or result.get("version_a") or {}


def _current_tree(review: dict) -> List[dict]:
    return _current_version(review).get("tree", []) or []


def _source_version(review: dict) -> str:
    result = review.get("result") or {}
    return (result.get("label_check_version")
            or review.get("label_a") or "现版配方")


def _leaf_ingredients(tree: List[dict]) -> List[str]:
    """配料树的全部叶子原料展示名（先序去重，同名叶子保留首次出现名）。

    匹配时按 :func:`normalize_name` 归一化去空白大小写，此处返回树中的
    原始名称作为唯一展示名。
    """
    seen = set()
    names: List[str] = []

    def walk(nodes: List[dict]) -> None:
        for node in nodes:
            children = node.get("children", []) or []
            if children:
                walk(children)
                continue
            name = node.get("name", "")
            key = normalize_name(name)
            if key and key not in seen:
                seen.add(key)
                names.append(name)

    walk(tree)
    return names


def _load_eligible_review(review_id) -> dict:
    """取出记录并校验选择资格，返回 (叶子原料展示名, 归一名→展示名)。"""
    try:
        rid = int(review_id)
    except (TypeError, ValueError):
        raise PoolTraceError(
            "记录 ID 无效，禁止溯源",
            problems=[_record_problem(review_id, "invalid",
                                      "记录 ID 无效，禁止选择")])
    review = db.get_review(rid)
    if review is None:
        raise PoolTraceError(
            "审查记录不存在或已删除，禁止溯源",
            problems=[_record_problem(rid, "missing",
                                      "审查记录不存在或已删除，禁止选择")])
    if review.get("needs_recheck"):
        reason = review.get("recheck_reason")
        raise PoolTraceError(
            "记录处于需复核状态，禁止溯源",
            problems=[_record_problem(
                review["id"], reason or "unknown",
                f"记录{blocked_message(reason)}，禁止选择",
                name=review["name"])])
    leaves = _leaf_ingredients(_current_tree(review))
    return {
        "id": review["id"],
        "name": review["name"],
        "source_version": _source_version(review),
        "leaves": leaves,
        "leaf_index": {normalize_name(n): n for n in leaves},
    }


def parse_pools(text: str, leaf_index: Dict[str, str]) -> List[dict]:
    """解析「池名：原料列表：阳性/阴性」逐行文本。

    任一行不合法即抛出 :class:`PoolTraceError`，problems 给出**全部**问题
    行（整批拒绝口径与别名批量导入、套餐规划一致：本次预览不产生任何结论）。
    返回 ``[{name, positive(bool), ingredients(排序去重展示名列表)}]``，
    保持录入顺序；原料引用按归一名匹配当前配方叶子原料。
    """
    pools: List[dict] = []
    problems: List[dict] = []
    seen: Dict[str, int] = {}
    for idx, raw_line in enumerate((text or "").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = _SPLIT_COLON_RE.split(line)
        if len(parts) != 3:
            problems.append({
                "line": idx, "type": "format",
                "message": "格式应为「池名：原料列表：阳性/阴性」"
                           "（需恰好 2 个冒号分隔 3 段）",
            })
            continue
        name = parts[0].strip()
        if not name:
            problems.append({"line": idx, "type": "empty_name",
                             "message": "池名不能为空"})
            continue
        if name in seen:
            problems.append({
                "line": idx, "type": "duplicate_name",
                "message": f"池名「{name}」与第 {seen[name]} 行重复，池名必须唯一",
            })
            continue
        # 名称即刻登记：即使本行后续（状态 / 原料）有问题，后面的重名行
        # 也要在同一批中被指出（全部问题一次列出）。
        seen[name] = idx
        status_text = parts[2].strip()
        if status_text not in _STATUS:
            problems.append({
                "line": idx, "type": "bad_status",
                "message": f"检测状态必须是「阳性」或「阴性」（收到 {status_text!r}）",
            })
            continue
        ingredients: List[str] = []
        unknown: List[str] = []
        local_seen = set()
        for token in _SPLIT_ITEMS_RE.split(parts[1]):
            token = token.strip()
            if not token:
                continue
            key = normalize_name(token)
            if key not in leaf_index:
                unknown.append(token)
            elif key not in local_seen:
                local_seen.add(key)
                ingredients.append(leaf_index[key])
        if unknown:
            problems.append({
                "line": idx, "type": "unknown_ingredient",
                "message": "以下原料不是该记录当前配方的叶子原料："
                           + "、".join(sorted(set(unknown)))
                           + "（仅可引用配方树中的叶子原料）",
            })
            continue
        pools.append({"name": name, "positive": _STATUS[status_text],
                      "ingredients": sorted(ingredients)})
    if problems:
        raise PoolTraceError(
            f"池化检测录入有 {len(problems)} 行未通过校验，请修正后重试",
            problems=problems)
    return pools


def solve_trace(leaves: List[str], pools: List[dict]) -> dict:
    """子集动态规划求命中全部阳性池的最少嫌疑原料集合。

    leaves 为当前配方全部叶子原料展示名；pools 为已解析的池行（保持录入
    顺序）。阴性池的并集先排除其全部原料；阳性池若不含任何未被排除的
    原料（含空原料列表）即冲突，返回 feasible=False 与全部冲突池。否则
    用 0/1 命中集 DP（状态位 = 阳性池下标）求最小集合大小，统计并列解数，
    再以后缀 DP + 按原料名升序贪心重建原料名序列字典序最小的唯一方案。
    """
    negatives = [p for p in pools if not p["positive"]]
    positives = [p for p in pools if p["positive"]]

    excluded = set()
    for pool in negatives:
        excluded.update(pool["ingredients"])

    conflicts = []
    for pool in positives:
        suspects = [i for i in pool["ingredients"] if i not in excluded]
        if not suspects:
            conflicts.append({
                "name": pool["name"],
                "ingredients": list(pool["ingredients"]),
                "excluded_ingredients": sorted(
                    set(pool["ingredients"]) & excluded),
            })

    overview = [{
        "name": p["name"],
        "positive": p["positive"],
        "ingredients": list(p["ingredients"]),
    } for p in pools]

    base = {
        "pools": overview,
        "positive_pools": [p["name"] for p in positives],
        "negative_pools": [p["name"] for p in negatives],
        "excluded_ingredients": sorted(excluded),
    }

    if conflicts:
        return {
            **base,
            "feasible": False,
            "conflict_pools": conflicts,
            "minimum_size": None,
            "optimal_count": 0,
            "suspects": [],
        }

    m = len(positives)
    full = (1 << m) - 1

    # 候选原料：出现在至少一个阳性池中且未被阴性证据排除；
    # 固定按原料名升序处理，保证并列键与方案唯一、可复现。
    cover: Dict[str, int] = {}
    for j, pool in enumerate(positives):
        for ingredient in pool["ingredients"]:
            if ingredient not in excluded:
                cover[ingredient] = cover.get(ingredient, 0) | (1 << j)
    candidates = sorted(cover)
    masks = [cover[name] for name in candidates]
    n = len(candidates)

    if m == 0:
        # 没有阳性池：空集合即解释全部（阴性池只用于排除）
        return {
            **base,
            "feasible": True,
            "conflict_pools": [],
            "minimum_size": 0,
            "optimal_count": 1,
            "suspects": [],
        }

    # 0/1 背包：dp[mask] = 恰好命中阳性池集合 mask 的（最少原料数, 并列
    # 集合数），不可达为 None。每个原料至多选一次，由「处理它之前」的
    # 快照转移；冗余原料（不带来新命中）保持 mask 不变，不产生新集合，
    # 直接跳过——最小命中集天然不含冗余原料。
    dp: List[Optional[Tuple[int, int]]] = [None] * (1 << m)
    dp[0] = (0, 1)
    for pmask in masks:
        snapshot = list(dp)
        for mask, state in enumerate(snapshot):
            if state is None:
                continue
            nmask = mask | pmask
            cand_size = state[0] + 1
            cur = dp[nmask]
            if cur is None or cand_size < cur[0]:
                dp[nmask] = (cand_size, state[1])
            elif cand_size == cur[0]:
                dp[nmask] = (cur[0], cur[1] + state[1])

    best = dp[full]
    # 冲突池已提前排除：每个阳性池至少含一个候选原料，full 必可达
    if best is None:  # pragma: no cover - 防御性分支
        return {
            **base,
            "feasible": False,
            "conflict_pools": [
                {"name": p["name"], "ingredients": list(p["ingredients"]),
                 "excluded_ingredients": sorted(
                     set(p["ingredients"]) & excluded)}
                for p in positives
                if not [i for i in p["ingredients"] if i not in excluded]],
            "minimum_size": None,
            "optimal_count": 0,
            "suspects": [],
        }
    minimum_size, ways = best

    # 后缀 DP：suf[k][mask] = 仅用 candidates[k:] 把已命中 mask 补到 full
    # 所需的最少追加原料数（不可达为 None）。随后按原料名升序贪心重建
    # 字典序最小的名称序列：已能以预算补全即停止（空续接是前缀，字典序
    # 最小）；否则取「加入后仍能以最少原料数补全」的名字最小原料——冗余
    # 原料无法满足预算恒等式而被自动排除。
    suf: List[List[Optional[int]]] = [
        [None] * (1 << m) for _ in range(n + 1)]
    suf[n][full] = 0
    for k in range(n - 1, -1, -1):
        pmask = masks[k]
        for mask in range(1 << m):
            skip = suf[k + 1][mask]
            tail = suf[k + 1][mask | pmask]
            take = tail + 1 if tail is not None else None
            if skip is None:
                chosen = take
            elif take is None:
                chosen = skip
            else:
                chosen = skip if skip < take else take
            suf[k][mask] = chosen

    chosen_names: List[str] = []
    hit = 0
    used = 0
    k = 0
    while not (hit == full and used == minimum_size):
        for j in range(k, n):
            tail = suf[j + 1][hit | masks[j]]
            if tail is not None and used + 1 + tail == minimum_size:
                chosen_names.append(candidates[j])
                hit |= masks[j]
                used += 1
                k = j + 1
                break
        else:  # pragma: no cover - 可行性已由正向 DP 保证，理论上不可达
            raise RuntimeError("最少嫌疑集合重建失败")

    positive_names = [p["name"] for p in positives]
    name_to_mask = {name: masks[i] for i, name in enumerate(candidates)}
    suspects = []
    for name in chosen_names:
        pmask = name_to_mask[name]
        explained = [positive_names[j] for j in range(m)
                     if pmask >> j & 1]
        suspects.append({
            "ingredient": name,
            "explains_pools": explained,
            "explains_count": len(explained),
        })
    return {
        **base,
        "feasible": True,
        "conflict_pools": [],
        "minimum_size": minimum_size,
        "optimal_count": ways,
        "suspects": suspects,
    }


def build_pool_trace_preview(review_id, text: str) -> dict:
    """阳性溯源预览的完整编排：校验记录资格后解析池行并求最小命中集（只读）。

    记录缺失或处于需复核状态时抛出 :class:`PoolTraceError`，problems 给出
    禁止选择的原因；池行解析问题（引用非配方叶子原料 / 池名重名 / 状态
    非法 / 格式错误）同样整批拒绝并逐行指出。
    """
    review = _load_eligible_review(review_id)
    pools = parse_pools(text, review["leaf_index"])
    if not pools:
        raise PoolTraceError(
            "至少录入一个检测池（每行「池名：原料列表：阳性/阴性」）",
            problems=[{"line": None, "type": "empty",
                       "message": "检测池内容为空（# 注释 / 空行不计入），请至少录入一行"}])
    result = solve_trace(review["leaves"], pools)
    result["review"] = {
        "id": review["id"],
        "name": review["name"],
        "source_version": review["source_version"],
        "leaf_ingredients": review["leaves"],
    }
    return result
