"""
Causal analysis utilities: DAG building, backdoor criterion,
adjustment set identification, collider/mediator detection.
"""
import networkx as nx
import pandas as pd
from typing import Set, Tuple, List


def build_dag(edges: pd.DataFrame) -> nx.DiGraph:
    G = nx.DiGraph()
    if edges.empty:
        return G
    for _, row in edges.iterrows():
        G.add_edge(str(row["from_var"]), str(row["to_var"]))
    return G


def get_node_roles(G: nx.DiGraph, theory_mapping: pd.DataFrame = None) -> dict:
    """
    Classify each node structurally:
      exogenous  — no parents
      outcome    — no children
      collider   — 2+ parents whose parents are unconnected
      mediator   — sits on a causal path
    Also uses proxy_type from theory_mapping if available.
    """
    proxy_map = {}
    if theory_mapping is not None and not theory_mapping.empty:
        if "proxy_type" in theory_mapping.columns:
            proxy_map = dict(zip(theory_mapping["var_name"], theory_mapping["proxy_type"]))

    collider_nodes = set(get_colliders(G))
    roles = {}
    for node in G.nodes():
        in_deg  = G.in_degree(node)
        out_deg = G.out_degree(node)
        pt = proxy_map.get(node, "")

        if node in collider_nodes:
            roles[node] = "collider"
        elif pt == "exogenous" or in_deg == 0:
            roles[node] = "exogenous"
        elif out_deg == 0:
            roles[node] = "outcome"
        else:
            roles[node] = "mediator"
    return roles


def get_colliders(G: nx.DiGraph) -> List[str]:
    """Nodes where two or more unconnected causes converge."""
    colliders = []
    for node in G.nodes():
        parents = list(G.predecessors(node))
        if len(parents) < 2:
            continue
        for i, p1 in enumerate(parents):
            for p2 in parents[i + 1:]:
                if not G.has_edge(p1, p2) and not G.has_edge(p2, p1):
                    colliders.append(node)
                    break
    return list(set(colliders))


def find_adjustment_set(
    edges: pd.DataFrame, X: str, Y: str
) -> Tuple[Set[str], Set[str], Set[str], str]:
    """
    Backdoor Criterion – find variables to control for to isolate X → Y.

    Returns
    -------
    adjustment_set   : variables to control
    mediators        : on-path variables (should NOT be controlled)
    descendants_X    : descendants of X (should NOT be controlled)
    message          : plain-language explanation
    """
    G = build_dag(edges)

    if X not in G.nodes():
        return set(), set(), set(), f"'{X}'이(가) DAG에 존재하지 않습니다."
    if Y not in G.nodes():
        return set(), set(), set(), f"'{Y}'이(가) DAG에 존재하지 않습니다."
    if not nx.has_path(G, X, Y):
        return set(), set(), set(), f"'{X}' → '{Y}' 인과 경로가 DAG에 없습니다."

    ancestors_X  = nx.ancestors(G, X)
    ancestors_Y  = nx.ancestors(G, Y)
    descendants_X = nx.descendants(G, X)
    parents_X    = set(G.predecessors(X))

    # Potential confounders: common ancestors of X and Y, excluding descendants of X
    confounders = (ancestors_X & ancestors_Y) - descendants_X - {X, Y}

    # Direct confounders: parents of X that also affect Y
    direct = parents_X & ancestors_Y - descendants_X

    # Minimal adjustment set: prefer direct (parents of X)
    adjustment_set = direct if direct else confounders

    # Mediators: descendants of X on the path to Y
    mediators = set()
    for node in descendants_X:
        if node != Y and nx.has_path(G, node, Y):
            mediators.add(node)

    if not adjustment_set:
        msg = (
            f"'{X}'에서 '{Y}'로의 경로에 교란변수가 감지되지 않습니다. "
            "통계적으로 충분한 엣지 정보가 없거나, 이미 단순 구조일 수 있습니다."
        )
    else:
        adj_str = ", ".join(f"**{v}**" for v in sorted(adjustment_set))
        med_str = (", ".join(f"**{v}**" for v in sorted(mediators))
                   if mediators else "없음")
        msg = (
            f"'{X}'이 '{Y}'에 미치는 **순수한 인과 효과**를 측정하려면 "
            f"{adj_str}을(를) 통제(고정)해야 합니다.\n\n"
            f"> ⚠️ **통제하면 안 되는 변수 (매개변수):** {med_str}  \n"
            f"> 매개변수를 통제하면 X → Y의 실제 효과가 차단됩니다."
        )

    return adjustment_set, mediators, descendants_X, msg


def identify_adjustment_set(
    edges: pd.DataFrame, X: str, Y: str, observed: Set[str], lang: str = "ko"
) -> dict:
    """
    Backdoor 식별 — 조정 기준(adjustment criterion, Perković et al. 2018)에 따라
    X → Y의 인과효과를 식별하기 위해 통제해야 할 '관측 가능한' 교란변수 집합을 찾는다.

    - 매개변수(mediator)와 그 후손은 통제 금지(forbidden)로 자동 제외.
    - 충돌부(collider)는 d-분리 검증으로 자동 제외(조건화 시 경로가 열리므로 무효 처리).
    - observed 에 없는 노드(잠재 구성개념)는 미관측으로 간주 → 조정집합에 넣을 수 없음.

    Parameters
    ----------
    edges     : 누적 DAG 엣지 (wl_edges_stat + wl_edges + wl_edges_latent_final)
    X, Y      : 처치(원인)·결과 변수
    observed  : 측정 가능한 변수 집합 (variable_meta 기준)

    Returns
    -------
    dict(status, adjustment_set, confounders, mediators, colliders,
         latent_blockers, paths, message)
        status ∈ {"no_node","no_path","identifiable","latent_confounded","no_backdoor"}
    """
    def _t(ko: str, en: str) -> str:
        return en if lang == "en" else ko

    G = build_dag(edges)

    base = dict(adjustment_set=set(), confounders=set(), mediators=set(),
                colliders=set(), latent_blockers=set(), paths=[])

    if X not in G.nodes() or Y not in G.nodes():
        return {**base, "status": "no_node",
                "message": _t(f"'{X}' 또는 '{Y}'이(가) 인과 그래프에 존재하지 않습니다.",
                              f"'{X}' or '{Y}' does not exist in the causal graph.")}

    causal_paths = list(nx.all_simple_paths(G, X, Y, cutoff=10))
    if not causal_paths:
        return {**base, "status": "no_path",
                "message": _t(f"'{X}' → '{Y}'로 향하는 인과 경로가 그래프에 없습니다. "
                              "두 변수 사이에 방향성 있는 직·간접 경로가 필요합니다.",
                              f"There is no causal path from '{X}' → '{Y}' in the graph. "
                              "A directed direct or indirect path between the two variables is required.")}

    # 적정 인과경로 노드(cn): X→…→Y 경로 위 노드(X 제외) = 매개변수 + Y
    cn = set()
    for p in causal_paths:
        cn.update(p)
    cn.discard(X)
    mediators = cn - {Y}

    # 통제 금지 집합: cn 과 그 후손 전체 + X
    forbidden = {X}
    for n in cn:
        forbidden.add(n)
        forbidden |= nx.descendants(G, n)

    # 정준 조정집합 후보: X·Y의 조상 중 forbidden 제외
    anc = nx.ancestors(G, X) | nx.ancestors(G, Y) | {X, Y}
    Z_can = (anc - forbidden) - {X, Y}
    Z_obs = {z for z in Z_can if z in observed}
    latent_in_can = Z_can - Z_obs

    # Proper backdoor graph: X에서 cn으로 나가는 '첫 인과 엣지' 제거
    Gp = G.copy()
    for m in cn:
        if Gp.has_edge(X, m):
            Gp.remove_edge(X, m)

    valid_obs = nx.is_d_separator(Gp, {X}, {Y}, Z_obs)
    valid_can = nx.is_d_separator(Gp, {X}, {Y}, Z_can)

    colliders = set(get_colliders(G))
    # 두 변수 모두의 조상 = 핵심 교란(공통원인)
    confounders = {z for z in Z_obs
                   if nx.has_path(G, z, X) and nx.has_path(G, z, Y)}

    common = dict(mediators=mediators, colliders=colliders, paths=causal_paths)

    if valid_obs:
        if Z_obs:
            adj_str = "、".join(sorted(Z_obs))
        else:
            adj_str = _t("없음 (조정 불필요)", "None (no adjustment needed)")
        med_str = "、".join(sorted(mediators)) if mediators else _t("없음", "None")
        msg = _t(
            f"✅ **식별 가능** — '{X}'→'{Y}'의 순수 인과효과를 보려면 "
            f"**{adj_str}**을(를) 통제(balancing)하면 됩니다.\n\n"
            f"> 🚫 통제 금지(매개변수): {med_str}\n"
            f"> 🚫 충돌부는 d-분리 검증으로 자동 제외되었습니다.",
            f"✅ **Identifiable** — to see the pure causal effect of '{X}'→'{Y}', "
            f"control (balance) for **{adj_str}**.\n\n"
            f"> 🚫 Do not control (mediators): {med_str}\n"
            f"> 🚫 Colliders were automatically excluded by d-separation checking."
        )
        return {**base, **common, "status": "identifiable",
                "adjustment_set": Z_obs, "confounders": confounders, "message": msg}

    if valid_can:
        # 정준집합으론 차단되나 그 안에 미관측 잠재노드가 필요 → 관측만으론 불가
        lat_str = "、".join(sorted(latent_in_can)) or _t("(미관측 공통원인)", "(unobserved common cause)")
        msg = _t(
            f"⚠️ **관측 변수만으로는 식별 불가** — backdoor 경로를 차단하려면 "
            f"미관측 잠재 구성개념 **{lat_str}**을(를) 통제해야 하지만 직접 측정되지 않습니다.\n\n"
            f"> 💜 해당 잠재요인의 proxy 변수를 추가 측정하거나 모형에 포함해야 "
            f"'{X}'→'{Y}' 효과를 식별할 수 있습니다.",
            f"⚠️ **Not identifiable from observed variables alone** — to block the backdoor path, "
            f"the unobserved latent construct **{lat_str}** must be controlled, but it is not directly measured.\n\n"
            f"> 💜 You must additionally measure a proxy variable for that latent factor or include it in the model "
            f"to identify the '{X}'→'{Y}' effect."
        )
        return {**base, **common, "status": "latent_confounded",
                "adjustment_set": Z_obs, "confounders": confounders,
                "latent_blockers": latent_in_can, "message": msg}

    msg = _t(
        f"❌ **유효한 backdoor 조정집합이 존재하지 않습니다** — 현재 그래프 구조에서는 "
        f"어떤 변수 집합으로도 '{X}'→'{Y}'의 모든 비인과 경로를 차단할 수 없습니다 "
        f"(예: 미관측 교란 또는 M-구조). frontdoor·도구변수 등 다른 식별 전략이 필요합니다.",
        f"❌ **No valid backdoor adjustment set exists** — in the current graph structure, "
        f"no set of variables can block all non-causal paths of '{X}'→'{Y}' "
        f"(e.g. unobserved confounding or an M-structure). Another identification strategy such as "
        f"frontdoor or instrumental variables is needed."
    )
    return {**base, **common, "status": "no_backdoor",
            "adjustment_set": Z_obs, "confounders": confounders, "message": msg}


def describe_path(G: nx.DiGraph, X: str, Y: str) -> List[List[str]]:
    """Return all simple directed paths from X to Y."""
    try:
        return list(nx.all_simple_paths(G, X, Y, cutoff=8))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []


def get_weak_edges(stat_edges: pd.DataFrame, threshold: float = 0.85) -> pd.DataFrame:
    """Return stat edges with low bootstrap strength."""
    if stat_edges.empty or "strength" not in stat_edges.columns:
        return pd.DataFrame()
    return stat_edges[stat_edges["strength"] < threshold].copy()


def match_latent_to_weak(
    weak_edges: pd.DataFrame, latent_edges: pd.DataFrame
) -> pd.DataFrame:
    """
    For each weak confirmed edge, find latent constructs that share
    at least one endpoint — these are candidate unobserved confounders.
    """
    if weak_edges.empty or latent_edges.empty:
        return pd.DataFrame()

    weak_nodes = set(weak_edges["from_var"]) | set(weak_edges["to_var"])
    mask = (
        latent_edges["to_var"].isin(weak_nodes) |
        latent_edges["from_var"].isin(weak_nodes)
    )
    return latent_edges[mask].copy()
