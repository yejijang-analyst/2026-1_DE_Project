"""
CausalDAG Explorer — Streamlit app
교육심리 연구자를 위한 인과 DAG 탐색 및 교란변수 분석 도구
"""
import os, tempfile
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import networkx as nx
from pyvis.network import Network

import db
import gemini_report
from causal_utils import (
    build_dag, get_node_roles, get_colliders,
    find_adjustment_set, identify_adjustment_set, describe_path,
    get_weak_edges, match_latent_to_weak,
)

# ── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CausalDAG Explorer",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── I18N ─────────────────────────────────────────────────────────────────────
# UI 텍스트 전용 다국어 헬퍼. 언어 선택은 사이드바 상단 셀렉트박스(key="lang")에서
# 이뤄지며 st.session_state["lang"] 에 "ko" / "en" 로 저장된다.
# DB에서 오는 내용(데이터셋명·변수 설명·이론 텍스트·Gemini 리포트)은 번역하지 않는다.
def t(ko: str, en: str) -> str:
    """현재 선택된 UI 언어에 맞는 문자열을 돌려준다 (기본값 한국어)."""
    return en if st.session_state.get("lang", "ko") == "en" else ko

# ── COLOR PALETTE ────────────────────────────────────────────────────────────
C = {
    "darkest":  "#1B4332",
    "dark":     "#2D6A4F",
    "medium":   "#40916C",
    "accent":   "#52B788",
    "soft":     "#74C69D",
    "light":    "#95D5B2",
    "lightest": "#D8F3DC",
    "bg":       "#F0F7F0",
    "white":    "#FFFFFF",
    "latent":   "#7B5EA7",      # purple for latent nodes
    "collider": "#E76F51",      # orange for colliders
    "warning":  "#F4A261",      # amber
    "danger":   "#D62828",      # red — blacklist 위반 엣지
    "gray":     "#B5C4B1",
    "text_dark": "#1B4332",
}

NODE_COLOR = {
    "exogenous": C["medium"],
    "mediator":  C["accent"],
    "outcome":   C["darkest"],
    "collider":  C["collider"],
    "latent":    C["latent"],
}

# ── GLOBAL CSS ───────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
/* Global */
html, body, [class*="css"] {{ font-family: 'Pretendard', 'Noto Sans KR', sans-serif; }}
.stApp {{ background: {C["bg"]}; }}

/* Sidebar */
[data-testid="stSidebar"] {{
    background: {C["darkest"]};
    color: {C["white"]};
}}
[data-testid="stSidebar"] * {{ color: {C["lightest"]} !important; }}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {{ color: {C["light"]} !important; }}

/* Cards */
.card {{
    background: {C["white"]};
    border-left: 4px solid {C["accent"]};
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 12px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07);
}}
.card-dark {{
    background: {C["darkest"]};
    color: {C["lightest"]};
    border-left: 4px solid {C["soft"]};
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 12px;
}}
.card-purple {{
    background: #F3F0F9;
    border-left: 4px solid {C["latent"]};
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 12px;
}}
.card-warn {{
    background: #FFF8F0;
    border-left: 4px solid {C["collider"]};
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 12px;
}}

/* Metric chips */
.chip {{
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.8em;
    font-weight: 600;
    margin: 2px;
}}
.chip-green  {{ background: {C["lightest"]}; color: {C["dark"]}; }}
.chip-purple {{ background: #EDE9F5; color: {C["latent"]}; }}
.chip-orange {{ background: #FEF0E6; color: {C["collider"]}; }}
.chip-gray   {{ background: #F0F4F0; color: #555; }}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {{
    background: {C["lightest"]};
    border-radius: 8px;
    padding: 4px;
    gap: 4px;
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 6px;
    color: {C["dark"]};
    font-weight: 500;
}}
.stTabs [aria-selected="true"] {{
    background: {C["dark"]} !important;
    color: {C["white"]} !important;
}}

/* Section headers */
.section-header {{
    color: {C["dark"]};
    font-size: 1.1em;
    font-weight: 700;
    border-bottom: 2px solid {C["light"]};
    padding-bottom: 6px;
    margin-bottom: 14px;
}}

/* Legend dots */
.dot {{
    display: inline-block;
    width: 12px; height: 12px;
    border-radius: 50%;
    margin-right: 6px;
    vertical-align: middle;
}}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS — PYVIS
# ══════════════════════════════════════════════════════════════════════════════
def _strength_to_width(s: float) -> float:
    return max(1.5, float(s) * 6)


def _pyvis_html(
    stat_edges: pd.DataFrame,
    candidate_edges: pd.DataFrame = None,
    latent_edges: pd.DataFrame = None,
    var_meta: pd.DataFrame = None,
    theory_map: pd.DataFrame = None,
    highlight_nodes: set = None,
    highlight_edges: set = None,     # set of (from,to) tuples
    red_edges: set = None,           # set of (from,to) tuples — blacklist 위반(빨강)
    rationale_map: dict = None,      # (from,to) -> LLM rationale 문자열
    show_stat: bool = True,
    show_candidate: bool = False,
    show_latent: bool = False,
    height: str = "520px",
    title: str = "",
) -> str:
    """Build pyvis Network and return HTML string."""

    net = Network(
        height=height, width="100%",
        directed=True, notebook=False,
        bgcolor=C["bg"],
    )

    # --- collect all nodes ---
    all_nodes: set = set()
    latent_only: set = set()

    if show_stat and stat_edges is not None and not stat_edges.empty:
        all_nodes |= set(stat_edges["from_var"]) | set(stat_edges["to_var"])
    if show_candidate and candidate_edges is not None and not candidate_edges.empty:
        all_nodes |= set(candidate_edges["from_var"]) | set(candidate_edges["to_var"])
    if show_latent and latent_edges is not None and not latent_edges.empty:
        latent_only = set(latent_edges["from_var"])
        all_nodes |= latent_only | set(latent_edges["to_var"])

    # variable meta
    var_desc: dict = {}
    if var_meta is not None and not var_meta.empty:
        var_desc = dict(zip(var_meta["var_name"], var_meta["description"]))

    # node roles from stat DAG
    ref_edges = stat_edges if (show_stat and stat_edges is not None and not stat_edges.empty) else (
        candidate_edges if candidate_edges is not None else pd.DataFrame()
    )
    G_ref = build_dag(ref_edges)
    roles = get_node_roles(G_ref, theory_map)

    # --- add nodes ---
    for node in all_nodes:
        is_latent = node in latent_only
        role = "latent" if is_latent else roles.get(node, "mediator")
        color = NODE_COLOR.get(role, C["accent"])

        if highlight_nodes and node in highlight_nodes:
            color = C["warning"]

        desc = var_desc.get(node, "")
        role_kr = {
            "exogenous": t("외생변수 (통제 불가)", "Exogenous (uncontrollable)"),
            "mediator":  t("매개변수 (중간 경로)", "Mediator (intermediate path)"),
            "outcome":   t("결과변수", "Outcome"),
            "collider":  t("충돌부 ⚠️", "Collider ⚠️"),
            "latent":    t("잠재 구성개념 (미관측)", "Latent construct (unobserved)"),
        }.get(role, role)

        tooltip = f"{node}\n{role_kr}"
        if desc:
            tooltip += f"\n{desc[:180]}"

        border = "#FFFFFF" if not (highlight_nodes and node in highlight_nodes) else C["collider"]
        net.add_node(
            node,
            label=node,
            title=tooltip,
            color={"background": color, "border": border,
                   "highlight": {"background": C["warning"], "border": C["collider"]}},
            size=22 if is_latent else 18,
            shape="ellipse" if not is_latent else "diamond",
            font={"size": 13, "color": C["white"],
                  "strokeWidth": 2, "strokeColor": C["darkest"]},
        )

    # --- add stat edges ---
    if show_stat and stat_edges is not None and not stat_edges.empty:
        for _, row in stat_edges.iterrows():
            key = (row["from_var"], row["to_var"])
            strength = float(row.get("strength", 1.0))
            is_red = bool(red_edges) and key in red_edges
            is_hi = highlight_edges and key in highlight_edges
            if is_red:
                color = C["danger"]
            elif is_hi:
                color = C["warning"]
            else:
                color = C["dark"]
            tip = f"{row['from_var']} → {row['to_var']}\n{t('부트스트랩 강도', 'Bootstrap strength')}: {strength:.2f}"
            if is_red:
                tip += t("\n⛔ blacklist 위반 (외생/결과 변수가 잘못된 위치)",
                         "\n⛔ blacklist violation (exogenous/outcome in wrong position)")
            rationale = (rationale_map or {}).get(key, "")
            if rationale:
                tip += f"\n{t('llm 판단 이유', 'LLM rationale')}: {rationale[:300]}"
            net.add_edge(
                row["from_var"], row["to_var"],
                title=tip,
                width=_strength_to_width(strength),
                color={"color": color, "highlight": C["warning"]},
                dashes=False,
            )

    # --- add candidate (uncertain) edges ---
    if show_candidate and candidate_edges is not None and not candidate_edges.empty:
        existing = set()
        if show_stat and stat_edges is not None and not stat_edges.empty:
            existing = {(r["from_var"], r["to_var"]) for _, r in stat_edges.iterrows()}
        for _, row in candidate_edges.iterrows():
            key = (row["from_var"], row["to_var"])
            if key in existing:
                continue
            rationale = str(row.get("rationale", ""))[:300]
            reason = str(row.get("reason", ""))   # uncertain_dir / weak_link
            tip = (
                f"{row['from_var']} → {row['to_var']}\n"
                f"[{reason}]\n"
                f"{t('llm 판단 이유', 'LLM rationale')}: {rationale}"
            )
            net.add_edge(
                row["from_var"], row["to_var"],
                title=tip,
                width=1.5,
                color={"color": C["gray"]},
                dashes=True,
            )

    # --- add latent edges ---
    if show_latent and latent_edges is not None and not latent_edges.empty:
        for _, row in latent_edges.iterrows():
            rationale = str(row.get("rationale", ""))[:300]
            tip = (
                f"{row['from_var']} → {row['to_var']}\n"
                f"[{t('잠재 구성개념', 'Latent construct')}]\n"
                f"{t('llm 판단 이유', 'LLM rationale')}: {rationale}"
            )
            net.add_edge(
                row["from_var"], row["to_var"],
                title=tip,
                width=1.5,
                color={"color": C["latent"]},
                dashes=True,
            )

    # --- physics options ---
    net.set_options("""{
      "physics": {
        "enabled": true,
        "barnesHut": {
          "gravitationalConstant": -9000,
          "centralGravity": 0.25,
          "springLength": 160,
          "springConstant": 0.04,
          "damping": 0.12
        },
        "minVelocity": 0.75
      },
      "edges": {
        "smooth": {"type": "curvedCW", "roundness": 0.15},
        "arrows": {"to": {"enabled": true, "scaleFactor": 0.7}}
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 80,
        "navigationButtons": true,
        "keyboard": true
      },
      "nodes": {
        "shadow": {"enabled": true, "size": 4, "x": 2, "y": 2}
      }
    }""")

    # save → read HTML
    tmp = os.path.join(tempfile.gettempdir(), "causal_dag_tmp.html")
    net.save_graph(tmp)
    with open(tmp, "r", encoding="utf-8") as f:
        html = f.read()

    # vis-network 9.x renders string titles as plain text — inject CSS so our
    # `\n` line breaks display cleanly (otherwise everything runs onto one line).
    tooltip_css = (
        "<style>div.vis-tooltip{white-space:pre-line!important;max-width:340px;"
        "font-family:'Pretendard','Noto Sans KR',sans-serif;font-size:12px;"
        "line-height:1.45;color:#1B4332;}</style>"
    )
    html = html.replace("</head>", tooltip_css + "</head>")
    return html


def _legend_html() -> str:
    items = [
        (C["medium"],  t("외생변수", "Exogenous")),
        (C["accent"],  t("매개변수", "Mediator")),
        (C["darkest"], t("결과변수", "Outcome")),
        (C["collider"],t("충돌부", "Collider")),
        (C["latent"],  t("잠재 구성개념", "Latent construct")),
        (C["warning"], t("선택된 변수", "Selected variable")),
    ]
    chips = "".join(
        f'<span class="dot" style="background:{col}"></span>{lbl}&nbsp;&nbsp;'
        for col, lbl in items
    )
    edge_items = [
        (C["dark"],   "━━", t("통계 확정 (굵기=강도)", "Statistically confirmed (width = strength)")),
        (C["danger"], "━━", t("blacklist 위반 (외생/결과 위치 오류)", "blacklist violation (exogenous/outcome misplaced)")),
        (C["gray"],   "┅┅", t("후보 엣지 (불확실)", "Candidate edge (uncertain)")),
        (C["latent"], "┅┅", t("잠재 구성개념", "Latent construct")),
    ]
    echips = "".join(
        f'<span style="color:{col};font-weight:700">{sym}</span> {lbl}&nbsp;&nbsp;'
        for col, sym, lbl in edge_items
    )
    return f"""
    <div style="background:{C['lightest']};border-radius:8px;padding:10px 14px;
                font-size:0.82em;margin-bottom:10px;">
      <b>{t("노드", "Nodes")}:</b> {chips}<br>
      <b>{t("엣지", "Edges")}:</b> {echips}
    </div>
    """


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # 언어 선택 — 다른 UI 텍스트보다 먼저 렌더해 session_state["lang"] 를 확정한다.
    st.selectbox(
        "🌐 Language / 언어",
        options=["ko", "en"],
        format_func=lambda c: {"ko": "한국어", "en": "English"}[c],
        key="lang",
    )

    st.markdown(f"""
    <div style="text-align:center;padding:12px 0 4px">
      <span style="font-size:2em">🌿</span>
      <h2 style="margin:4px 0;font-size:1.2em;color:{C['light']}">CausalDAG Explorer</h2>
      <p style="font-size:0.78em;color:{C['gray']}">{t("교육심리 인과 구조 분석 도구", "Educational-psychology causal structure explorer")}</p>
    </div>
    <hr style="border-color:{C['dark']};margin:8px 0">
    """, unsafe_allow_html=True)

    dataset = st.selectbox(
        t("📦 데이터셋 선택", "📦 Select dataset"),
        options=list(db.DATASET_LABELS.keys()),
        format_func=lambda k: db.DATASET_LABELS[k],
    )

    # Domain info
    domain = db.load_domain_config(dataset)
    _env_map = {
        "classroom": t("교실 학습", "Classroom learning"),
        "game":      t("게임 환경", "Game environment"),
    }
    _learn_map = {
        "student": t("학생", "Student"),
        "player":  t("플레이어", "Player"),
    }
    env_kr   = _env_map.get(domain.get("learning_env",""), domain.get("learning_env",""))
    learn_kr = _learn_map.get(domain.get("learner_type",""), domain.get("learner_type",""))

    st.markdown(f"""
    <div style="background:{C['dark']};border-radius:8px;padding:12px;margin-top:8px">
      <div style="font-size:0.8em;color:{C['light']}">{t("학습 환경", "Learning environment")}</div>
      <div style="font-weight:600;color:{C['white']}">{env_kr}</div>
      <div style="font-size:0.8em;color:{C['light']};margin-top:6px">{t("학습자 유형", "Learner type")}</div>
      <div style="font-weight:600;color:{C['white']}">{learn_kr}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#2D6A4F;margin:12px 0'>", unsafe_allow_html=True)

    # Variable list
    vm = db.load_variable_meta(dataset)
    if not vm.empty:
        st.markdown(f"<div style='font-size:0.85em;color:{C['light']};font-weight:600'>{t('📋 변수 목록', '📋 Variables')} ({len(vm)}{t('개', '')})</div>", unsafe_allow_html=True)
        for _, row in vm.iterrows():
            st.markdown(
                f"<div style='font-size:0.75em;color:{C['gray']};padding:2px 0'>"
                f"<b style='color:{C['light']}'>{row['var_name']}</b><br>"
                f"<span style='padding-left:8px'>{str(row['description'])[:60]}{'…' if len(str(row['description']))>60 else ''}</span>"
                f"</div>",
                unsafe_allow_html=True
            )

    st.markdown("<hr style='border-color:#2D6A4F;margin:12px 0'>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:0.72em;color:{C['gray']};text-align:center'>{t('엣지에 마우스를 올리면<br>LLM 판단 근거가 표시됩니다', 'Hover over an edge to<br>see the LLM rationale')}</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=300)
def load_all(ds):
    return {
        "stat":      db.load_wl_edges_stat(ds),
        "stat_raw":  db.load_wl_edges_stat_raw(ds),
        "wl":        db.load_wl_edges(ds),
        "latent":    db.load_wl_edges_latent_final(ds),
        "bl":        db.load_bl_edges(ds),
        "vm":        db.load_variable_meta(ds),
        "tm":        db.load_theory_mapping(ds),
        "th_meta":   db.load_theory_meta(ds),
        "interp":    db.load_interpretation(ds),
        "domain":    db.load_domain_config(ds),
        "rationale": db.load_edge_rationales(ds),
    }

data = load_all(dataset)
stat_edges    = data["stat"]
stat_raw_edges = data["stat_raw"]
wl_edges      = data["wl"]
latent_edges  = data["latent"]
bl_edges      = data["bl"]
var_meta      = data["vm"]
theory_map    = data["tm"]
theory_meta   = data["th_meta"]
interp        = data["interp"]
rationale_map = data["rationale"]

# 통계 DAG는 wl_edges_stat_raw(블랙리스트 필터 이전)를 표시한다.
# raw 엣지 중 bl_edges에 걸린(외생/결과 변수 위치 오류) 엣지 = 빨강 강조 대상.
stat_display_edges = stat_raw_edges if not stat_raw_edges.empty else stat_edges
bl_edge_set = (
    {(r["from_var"], r["to_var"]) for _, r in bl_edges.iterrows()}
    if not bl_edges.empty else set()
)
red_edge_set = {
    (r["from_var"], r["to_var"])
    for _, r in stat_display_edges.iterrows()
    if (r["from_var"], r["to_var"]) in bl_edge_set
} if not stat_display_edges.empty else set()

# 누적 인과 그래프 (wl_edges_stat + wl_edges + wl_edges_latent_final) — backdoor 식별용.
# variable_meta에 있는 변수 = 관측 가능, 잠재 구성개념 = 미관측으로 처리.
_cc = ["from_var", "to_var"]
cumulative_edges = pd.concat(
    [d[_cc] for d in (stat_edges, wl_edges, latent_edges) if not d.empty]
    or [pd.DataFrame(columns=_cc)],
    ignore_index=True,
).drop_duplicates()
observed_vars = set(var_meta["var_name"]) if not var_meta.empty else set()
outcome_vars = (
    theory_map.loc[theory_map["proxy_type"] == "outcome", "var_name"].tolist()
    if not theory_map.empty and "proxy_type" in theory_map.columns else []
)

# summary stats
n_stat   = len(stat_display_edges)   # 통계 DAG = wl_edges_stat_raw
n_red    = len(red_edge_set)         # 그중 blacklist 위반(빨강)
n_cand   = len(wl_edges)
n_latent = len(latent_edges)
all_vars = sorted(
    set(stat_edges["from_var"].tolist() + stat_edges["to_var"].tolist()) if not stat_edges.empty
    else set(wl_edges["from_var"].tolist() + wl_edges["to_var"].tolist()) if not wl_edges.empty
    else []
)

# header
st.markdown(f"""
<div style="background:{C['dark']};color:{C['white']};
            padding:18px 24px;border-radius:12px;margin-bottom:18px;
            display:flex;align-items:center;gap:16px">
  <span style="font-size:2em">🌿</span>
  <div>
    <h1 style="margin:0;font-size:1.5em">CausalDAG Explorer</h1>
    <p style="margin:2px 0 0;opacity:0.8;font-size:0.9em">
      {db.DATASET_LABELS[dataset]} &nbsp;|&nbsp;
      {t("통계 DAG 엣지", "Statistical DAG edges")} <b>{n_stat}</b> ({t("blacklist 위반", "blacklist violations")} <b>{n_red}</b>) &nbsp;·&nbsp; {t("후보 엣지", "Candidate edges")} <b>{n_cand}</b> &nbsp;·&nbsp; {t("잠재 요인", "Latent factors")} <b>{n_latent}</b>
    </p>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    t("🗺️ DAG 탐색", "🗺️ DAG Explorer"),
    t("📖 인과 해석", "📖 Causal Interpretation"),
    t("🎯 교란변수 식별", "🎯 Confounder Identification"),
    t("🔍 잠재 요인 탐색", "🔍 Latent Factors"),
    t("💬 연구자 질의", "💬 Researcher Query"),
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DAG 탐색
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown(f'<div class="section-header">{t("인과 DAG 시각화", "Causal DAG Visualization")}</div>', unsafe_allow_html=True)

    MODE_STAT = t("통계 DAG", "Statistical DAG")
    MODE_COMPARE = t("후보 DAG 비교", "Compare candidate DAG")
    MODE_LATENT = t("잠재 구성개념 포함", "Include latent constructs")

    col_opt1, col_opt2, col_opt3 = st.columns(3)
    with col_opt1:
        show_mode = st.radio(
            t("표시 모드", "Display mode"),
            [MODE_STAT, MODE_COMPARE, MODE_LATENT],
            horizontal=False,
        )
    with col_opt2:
        st.markdown(f"""
        <div style="background:{C['lightest']};border-radius:8px;padding:12px;font-size:0.85em">
          <b>{MODE_STAT}</b><br>{t("부트스트랩으로 확정된 인과 엣지만 표시", "Shows only bootstrap-confirmed causal edges")}<br><br>
          <b>{MODE_COMPARE}</b><br>{t("LLM 제안 전체 엣지와 나란히 비교", "Side-by-side with all LLM-proposed edges")}<br><br>
          <b>{t("잠재 포함", "Include latent")}</b><br>{t("이론 기반 미관측 구성개념까지 포함", "Adds theory-based unobserved constructs")}
        </div>
        """, unsafe_allow_html=True)
    with col_opt3:
        show_legend = st.checkbox(t("범례 표시", "Show legend"), value=True)
        highlight_bl = st.toggle(
            t("⛔ blacklist 위반 엣지 빨강 표시", "⛔ Mark blacklist-violating edges in red"),
            value=False,
            help=t("wl_edges_stat에 남은 엣지는 진한 초록, "
                   "bl_edges에 걸려 제거 대상이 된(외생/결과 변수가 잘못된 위치) 엣지는 빨강으로 표시합니다.",
                   "Edges kept in wl_edges_stat are dark green; edges caught by bl_edges "
                   "(exogenous/outcome variable in the wrong position) are shown in red."),
            disabled=(len(red_edge_set) == 0),
        )

    if show_legend:
        st.markdown(_legend_html(), unsafe_allow_html=True)

    active_red = red_edge_set if highlight_bl else None

    if stat_display_edges.empty:
        st.warning(t("wl_edges_stat_raw 테이블이 비어 있습니다. Colab 파이프라인을 먼저 실행해주세요.",
                     "The wl_edges_stat_raw table is empty. Run the Colab pipeline first."))
    else:
        if show_mode == MODE_STAT:
            html = _pyvis_html(
                stat_edges=stat_display_edges,
                var_meta=var_meta, theory_map=theory_map,
                rationale_map=rationale_map,
                red_edges=active_red,
                show_stat=True,
            )
            if highlight_bl and red_edge_set:
                st.caption(t(
                    f"✅ 통계 DAG 엣지 {n_stat}개 (wl_edges_stat_raw) | "
                    f"진한 초록 = wl_edges_stat {n_stat - n_red}개 · "
                    f"빨강 = blacklist 위반 {n_red}개 | 두께 = 부트스트랩 강도",
                    f"✅ {n_stat} statistical DAG edges (wl_edges_stat_raw) | "
                    f"dark green = wl_edges_stat ({n_stat - n_red}) · "
                    f"red = blacklist violations ({n_red}) | width = bootstrap strength"
                ))
            else:
                st.caption(t(
                    f"✅ 부트스트랩 통계 DAG 엣지 {n_stat}개 | 엣지 두께 = 부트스트랩 강도",
                    f"✅ {n_stat} bootstrap statistical DAG edges | edge width = bootstrap strength"
                ))
            components.html(html, height=540, scrolling=False)

        elif show_mode == MODE_COMPARE:
            # 누적 1단계: wl_edges_stat (bl 교정 완료) + wl_edges (LLM 추가 제안)
            n_stat_final = len(stat_edges)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"<div style='text-align:center;font-weight:700;color:{C['dark']}'>{t('📊 통계 DAG (bl 교정 후)', '📊 Statistical DAG (after bl correction)')}</div>", unsafe_allow_html=True)
                h1 = _pyvis_html(stat_edges=stat_edges, var_meta=var_meta, theory_map=theory_map,
                                 rationale_map=rationale_map,
                                 show_stat=True, height="440px")
                components.html(h1, height=450, scrolling=False)
                st.caption(t(f"wl_edges_stat 엣지 {n_stat_final}개", f"{n_stat_final} wl_edges_stat edges"))
            with c2:
                st.markdown(f"<div style='text-align:center;font-weight:700;color:{C['dark']}'>{t('🤖 LLM 교정 DAG = wl_edges_stat + wl_edges', '🤖 LLM-corrected DAG = wl_edges_stat + wl_edges')}</div>", unsafe_allow_html=True)
                h2 = _pyvis_html(stat_edges=stat_edges, candidate_edges=wl_edges,
                                 var_meta=var_meta, theory_map=theory_map,
                                 rationale_map=rationale_map,
                                 show_stat=True, show_candidate=True, height="440px")
                components.html(h2, height=450, scrolling=False)
                st.caption(t(f"통계 {n_stat_final}개 + LLM 추가 제안 (점선)", f"{n_stat_final} statistical + LLM-added proposals (dashed)"))

            # wl_edges 표 — LLM DAG 교정 근거
            if not wl_edges.empty:
                tbl = wl_edges[["from_var","to_var","reason","rationale"]].copy()
                tbl.columns = [t("원인", "Cause"), t("결과", "Effect"), t("유형", "Type"), t("LLM 판단 이유", "LLM rationale")]
                with st.expander(t(f"📑 wl_edges — LLM 추가 제안 엣지 ({len(tbl)}개) · uncertain_dir(방향 지정) / weak_link(유지·제거 판단)",
                                   f"📑 wl_edges — LLM-added proposed edges ({len(tbl)}) · uncertain_dir (direction) / weak_link (keep/drop)")):
                    st.dataframe(tbl, use_container_width=True, height=240)

        else:  # 잠재 포함 — 누적 2단계
            # wl_edges_stat + wl_edges + wl_edges_latent_final
            n_stat_final = len(stat_edges)
            html = _pyvis_html(
                stat_edges=stat_edges, candidate_edges=wl_edges, latent_edges=latent_edges,
                var_meta=var_meta, theory_map=theory_map,
                rationale_map=rationale_map,
                show_stat=True, show_candidate=True, show_latent=True, height="560px",
            )
            st.caption(t(
                f"✅ 누적 DAG = wl_edges_stat {n_stat_final}개 + wl_edges {n_cand}개 "
                f"+ 💜 wl_edges_latent_final {n_latent}개 (점선·마름모 노드)",
                f"✅ Cumulative DAG = wl_edges_stat ({n_stat_final}) + wl_edges ({n_cand}) "
                f"+ 💜 wl_edges_latent_final ({n_latent}) (dashed / diamond nodes)"
            ))
            components.html(html, height=575, scrolling=False)

            # wl_edges_latent_final 표 — 잠재변수 통한 추가 교정 근거
            if not latent_edges.empty:
                ltbl = latent_edges[["from_var","to_var","rationale"]].copy()
                ltbl.columns = [t("잠재 구성개념", "Latent construct"), t("관측 변수", "Observed variable"), t("LLM 판단 이유", "LLM rationale")]
                with st.expander(t(f"💜 wl_edges_latent_final — 잠재변수 통한 추가 교정 ({len(ltbl)}개)",
                                   f"💜 wl_edges_latent_final — additional correction via latent variables ({len(ltbl)})")):
                    st.dataframe(ltbl, use_container_width=True, height=240)

        # edge detail table
        with st.expander(t("📋 통계 DAG 엣지 상세 보기 (wl_edges_stat_raw)", "📋 Statistical DAG edge details (wl_edges_stat_raw)")):
            if not stat_display_edges.empty:
                disp = stat_display_edges.copy()
                disp[t("blacklist 위반", "blacklist violation")] = disp.apply(
                    lambda r: t("⛔ 예", "⛔ Yes") if (r["from_var"], r["to_var"]) in red_edge_set else "",
                    axis=1,
                )
                disp = disp.rename(columns={"from_var":t("원인 변수", "Cause variable"),"to_var":t("결과 변수", "Effect variable"),
                                            "strength":t("부트스트랩 강도", "Bootstrap strength"),"tag":t("상태", "Status")})
                st.dataframe(disp, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — 인과 해석
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown(f'<div class="section-header">{t("AI 생성 인과 해석 리포트", "AI-generated Causal Interpretation Report")}</div>', unsafe_allow_html=True)

    sess_key = f"interp_api_{dataset}"
    has_key = gemini_report.get_api_key() is not None
    cached = st.session_state.get(sess_key)

    # 미리 구워 둔 DB 저장본(dag_interpretations)이 있으면 오프라인 데모에서도 바로 표시
    has_saved = bool(interp)

    col_gen, col_info = st.columns([1, 2])
    with col_gen:
        if cached or has_saved:
            # 저장본/캐시가 있으면 기본은 그걸 사용, '다시 생성'을 눌러야만 API 재호출
            do_generate = st.button(
                t("🔄 다시 생성 (API 재호출)", "🔄 Regenerate (call API again)"),
                use_container_width=True, disabled=not has_key, key="regen_interp",
            )
        else:
            do_generate = st.button(
                t("🤖 Gemini API로 해석 생성", "🤖 Generate interpretation via Gemini API"), type="primary",
                use_container_width=True, disabled=not has_key, key="gen_interp",
            )
    with col_info:
        if cached:
            st.caption(t(f"✅ 캐시된 해석을 표시 중입니다 (API 호출 없음). 새로 만들려면 **🔄 다시 생성**을 누르세요. · 모델 {gemini_report.MODEL_ID}",
                         f"✅ Showing cached interpretation (no API call). Press **🔄 Regenerate** to rebuild. · Model {gemini_report.MODEL_ID}"))
        elif has_saved:
            st.caption(t("✅ 저장된 해석을 표시 중입니다 (API 호출 없음).",
                         "✅ Showing saved interpretation (no API call)."))
        elif not has_key:
            st.caption(t("⚠️ GEMINI_API_KEY가 설정되지 않았습니다. `.streamlit/secrets.toml`에 키를 넣어주세요.",
                         "⚠️ GEMINI_API_KEY is not set. Add the key to `.streamlit/secrets.toml`."))
        else:
            st.caption(t(f"AI Studio 키로 **{db.DATASET_LABELS[dataset]}**의 간결 해석·이론 기반 심층 해석을 실시간 생성합니다. (모델: {gemini_report.MODEL_ID})",
                         f"Generates a brief and a theory-based in-depth interpretation of **{db.DATASET_LABELS[dataset]}** live using your AI Studio key. (Model: {gemini_report.MODEL_ID})"))

    if do_generate:
        with st.spinner(t("🤖 Gemini API 추론 중... 잠시만 기다려주세요 (최대 ~30초)",
                          "🤖 Running Gemini API inference... please wait (up to ~30s)")):
            try:
                st.session_state[sess_key] = gemini_report.generate_interpretation(dataset)
                st.success(t("✅ 해석 생성 완료", "✅ Interpretation generated"))
                cached = st.session_state[sess_key]
            except RuntimeError as e:
                st.error(str(e))
            except Exception as e:
                st.error(t(f"❌ 해석 생성 실패: {e}", f"❌ Failed to generate interpretation: {e}"))
                raw = getattr(e, "raw", None)
                if raw:
                    with st.expander(t("🔎 RAW 응답 보기 (JSON 파싱 실패)", "🔎 View RAW response (JSON parse failed)")):
                        st.code(raw)

    # API로 생성한 결과가 있으면 우선, 없으면 DB 적재본(temp_report 등)
    active_interp = st.session_state.get(sess_key) or interp

    if not active_interp:
        st.markdown(f"""
        <div class="card-warn">
          <b>{t("⚠️ 아직 해석이 없습니다", "⚠️ No interpretation yet")}</b><br>
          {t("위의 <b>🤖 Gemini API로 해석 생성</b> 버튼을 누르면 현재 데이터셋의 간결 해석(brief)과 이론 기반 심층 해석(rich)을 즉시 생성합니다.",
             "Press the <b>🤖 Generate interpretation via Gemini API</b> button above to instantly generate a brief and a theory-based in-depth (rich) interpretation for the current dataset.")}
        </div>
        """, unsafe_allow_html=True)
    else:
        src = active_interp.get("_source", "")
        if src == "gemini_api":
            src_badge = t(f" · ✨ Gemini API 실시간 생성 ({active_interp.get('_model','')})",
                          f" · ✨ Generated live via Gemini API ({active_interp.get('_model','')})")
        elif src == "dag_interpretations":
            src_badge = t(" · 💾 저장된 리포트(dag_interpretations)", " · 💾 Saved report (dag_interpretations)")
        else:
            src_badge = ""
        created_at = active_interp.get("created_at", "")
        st.caption(f"🕐 {created_at}{src_badge}")

        # Brief
        brief = str(active_interp.get("brief", "")).replace("\n", "<br>")
        if brief:
            st.markdown(f"""
            <div class="card">
              <div class="section-header">{t("📌 핵심 구조 해석", "📌 Core Structural Interpretation")}</div>
              {brief}
            </div>
            """, unsafe_allow_html=True)

        # Rich
        rich = str(active_interp.get("rich", "")).replace("\n", "<br>")
        if rich:
            st.markdown(f"""
            <div class="card">
              <div class="section-header">{t("📚 이론 기반 심층 해석", "📚 Theory-based In-depth Interpretation")}</div>
              {rich}
            </div>
            """, unsafe_allow_html=True)

        # 보조 필드(temp_report 등 추가 키가 있을 때만 표시)
        cw = active_interp.get("collider_warn", "") or active_interp.get("collider_warning", "")
        mn = active_interp.get("mediator_note", "")
        pg = active_interp.get("proxy_guide", "")
        if cw or mn or pg:
            col_a, col_b = st.columns(2)
            with col_a:
                if cw:
                    st.markdown(f"""
                    <div class="card-warn">
                      <b>{t("⚡ 충돌부(Collider) 주의사항", "⚡ Collider caveats")}</b><br>{cw}
                    </div>""", unsafe_allow_html=True)
                if mn:
                    st.markdown(f"""
                    <div class="card">
                      <b>{t("🔗 매개변수 과도 통제 위험", "🔗 Risk of over-controlling mediators")}</b><br>{mn}
                    </div>""", unsafe_allow_html=True)
            with col_b:
                if pg:
                    st.markdown(f"""
                    <div class="card-purple">
                      <b>{t("💜 잠재 구성개념별 Proxy 변수 안내", "💜 Proxy variables per latent construct")}</b><br>{pg}
                    </div>""", unsafe_allow_html=True)

    # Theory meta table
    if not theory_meta.empty:
        with st.expander(t("📖 이론 프레임워크 전체 보기", "📖 View full theory framework")):
            st.dataframe(theory_meta, use_container_width=True, height=200)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — 교란변수 식별 (D-separation / Backdoor Criterion)
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown(f'<div class="section-header">{t("🎯 교란변수 식별 — 무엇을 통제해야 하는가?", "🎯 Confounder Identification — What Should Be Controlled?")}</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="card" style="font-size:0.9em">
      <b>{t("📘 교란변수(Confounder)란?", "📘 What is a confounder?")}</b><br>
      {t("원인 변수(X)와 결과 변수(Y) 모두에 영향을 주는 변수입니다. 통제하지 않으면 X → Y의 효과가 왜곡되어 측정됩니다.",
         "A variable that affects both the cause (X) and the effect (Y). If not controlled, the X → Y effect is measured with distortion.")}<br><br>
      {t("<b>조정 기준(Adjustment Criterion · Perković et al. 2018)</b>을 누적 DAG <code>(wl_edges_stat + wl_edges + wl_edges_latent_final)</code>에 적용해 X → Y의 <u>순수 인과효과</u>를 식별하기 위한 통제 집합을 d-분리로 자동 계산합니다. <b>매개변수·충돌부는 자동 제외</b>되고, 미관측 잠재 구성개념이 교란을 일으키면 <u>식별 불가</u>로 표시됩니다.",
         "Applies the <b>Adjustment Criterion (Perković et al. 2018)</b> to the cumulative DAG <code>(wl_edges_stat + wl_edges + wl_edges_latent_final)</code> and automatically computes the control set that identifies the <u>pure causal effect</u> of X → Y via d-separation. <b>Mediators and colliders are excluded automatically</b>, and if an unobserved latent construct causes confounding it is marked <u>not identifiable</u>.")}
    </div>
    """, unsafe_allow_html=True)

    if cumulative_edges.empty:
        st.warning(t("누적 DAG 엣지가 없어 분석을 수행할 수 없습니다.", "No cumulative DAG edges — analysis cannot be performed."))
    elif not outcome_vars:
        st.warning(t("theory_mapping에 proxy_type='outcome' 변수가 없어 결과 변수를 지정할 수 없습니다.",
                     "No proxy_type='outcome' variable in theory_mapping, so an outcome variable cannot be set."))
    else:
        explanatory_vars = sorted(observed_vars) if observed_vars else all_vars
        col_x, col_y, col_btn = st.columns([1, 1, 0.5])
        with col_x:
            exposure = st.selectbox(
                t("🔵 원인 변수 (X) — 설명변수 전체", "🔵 Cause variable (X) — all explanatory vars"), explanatory_vars, key="exp",
            )
        with col_y:
            outcome = st.selectbox(
                t("🔴 결과 변수 (Y) — proxy_type='outcome'", "🔴 Outcome variable (Y) — proxy_type='outcome'"), outcome_vars, key="out",
                help=t("theory_mapping에서 결과(outcome)로 지정된 변수만 선택 가능합니다.",
                       "Only variables designated as outcome in theory_mapping can be selected."),
            )
        with col_btn:
            st.write("")
            st.write("")
            run_adj = st.button(t("▶ 식별 실행", "▶ Run identification"), type="primary", use_container_width=True)

        if run_adj:
            res = identify_adjustment_set(cumulative_edges, exposure, outcome, observed_vars,
                                          lang=st.session_state.get("lang", "ko"))
            status = res["status"]
            adj_set = res["adjustment_set"]
            mediators = res["mediators"]
            confounders = res["confounders"]
            latent_blockers = res["latent_blockers"]

            # Summary chips
            chip_html = ""
            if adj_set:
                chip_html += "".join(
                    f'<span class="chip chip-orange">{t("통제(balancing)", "Control (balancing)")}: {v}</span>' for v in sorted(adj_set)
                )
            if mediators:
                chip_html += "".join(
                    f'<span class="chip chip-green">{t("통제 금지(매개)", "Do not control (mediator)")}: {v}</span>' for v in sorted(mediators)
                )
            if latent_blockers:
                chip_html += "".join(
                    f'<span class="chip chip-purple">{t("미관측 교란", "Unobserved confounder")}: {v}</span>' for v in sorted(latent_blockers)
                )
            if not chip_html:
                chip_html = f'<span class="chip chip-gray">{t("조정 불필요 (교란 없음)", "No adjustment needed (no confounding)")}</span>'
            st.markdown(f"<div style='margin:10px 0'>{chip_html}</div>", unsafe_allow_html=True)

            # Main message
            st.markdown(f"""
            <div class="card-dark" style="font-size:0.95em">
              <b>{t("📊 식별 결과", "📊 Identification result")}</b><br><br>
              {res['message']}
            </div>
            """, unsafe_allow_html=True)

            # Plain-language box
            if status == "identifiable" and adj_set:
                adj_plain = "、".join(sorted(adj_set))
                st.markdown(t(f"""
                <div class="card" style="font-size:0.9em">
                  <b>🗣️ 쉬운 설명 (비전공자용)</b><br><br>
                  "<b>{exposure}</b>가 <b>{outcome}</b>에 미치는 영향"만을 정확히 보려면,
                  분석 시 <b>{adj_plain}</b>의 값이 같은 그룹끼리 비교(또는 통계 모형에 포함)해야 합니다.
                </div>
                """,
                f"""
                <div class="card" style="font-size:0.9em">
                  <b>🗣️ Plain-language explanation</b><br><br>
                  To isolate "the effect of <b>{exposure}</b> on <b>{outcome}</b>",
                  compare groups that share the same values of <b>{adj_plain}</b> (or include them in your statistical model).
                </div>
                """), unsafe_allow_html=True)
            elif status == "latent_confounded":
                lat_plain = "、".join(sorted(latent_blockers)) or t("미관측 공통원인", "unobserved common cause")
                st.markdown(t(f"""
                <div class="card-purple" style="font-size:0.9em">
                  <b>💜 식별 불가 — 미관측 교란</b><br><br>
                  관측 변수만으로는 <b>{exposure}</b> → <b>{outcome}</b> 효과를 분리할 수 없습니다.
                  <b>{lat_plain}</b>이(가) 두 변수의 공통원인이지만 직접 측정되지 않기 때문입니다.
                  해당 잠재요인의 <b>proxy 변수</b>를 추가로 측정해 모형에 포함하면 식별이 가능해집니다.
                </div>
                """,
                f"""
                <div class="card-purple" style="font-size:0.9em">
                  <b>💜 Not identifiable — unobserved confounding</b><br><br>
                  The <b>{exposure}</b> → <b>{outcome}</b> effect cannot be isolated from observed variables alone,
                  because <b>{lat_plain}</b> is a common cause of both variables but is not directly measured.
                  Identification becomes possible if you additionally measure a <b>proxy variable</b> for that latent factor and include it in the model.
                </div>
                """), unsafe_allow_html=True)

            # Directed (causal) paths
            paths = res["paths"]
            if paths:
                with st.expander(t(f"🛤️ '{exposure}' → '{outcome}' 인과 경로 ({len(paths)}개)",
                                   f"🛤️ '{exposure}' → '{outcome}' causal paths ({len(paths)})")):
                    for p in paths:
                        st.markdown(f"- {' → '.join(p)}")

            # Highlighted DAG (누적 DAG에서 강조)
            highlight = set(adj_set) | {exposure, outcome}
            h_dag = _pyvis_html(
                stat_edges=stat_edges, candidate_edges=wl_edges, latent_edges=latent_edges,
                var_meta=var_meta, theory_map=theory_map,
                rationale_map=rationale_map,
                show_stat=True, show_candidate=True, show_latent=True,
                highlight_nodes=highlight,
                height="480px",
            )
            st.caption(t("주황색 강조 = 분석 대상(X·Y) 및 통제 변수 | 점선=후보·잠재 엣지",
                         "Orange highlight = analysis targets (X·Y) and control variables | dashed = candidate/latent edges"))
            components.html(h_dag, height=490, scrolling=False)

        # Reference table — confounder relations across cumulative DAG
        with st.expander(t("📋 전체 변수 교란 관계 참조표 (누적 DAG)", "📋 Confounding relations reference table (cumulative DAG)")):
            G_cum = build_dag(cumulative_edges)
            rows = []
            for v in sorted(G_cum.nodes()):
                parents = list(G_cum.predecessors(v))
                ancestors = nx.ancestors(G_cum, v)
                rows.append({
                    t("변수", "Variable"): v,
                    t("관측", "Observation"): t("관측", "Observed") if v in observed_vars else t("💜 잠재(미관측)", "💜 Latent (unobserved)"),
                    t("직접 원인 (부모)", "Direct causes (parents)"): ", ".join(parents) or t("없음", "None"),
                    t("조상 변수 수", "# ancestors"): len(ancestors),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — 잠재 요인 탐색
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown(f'<div class="section-header">{t("🔍 잠재 요인 & 미관측 교란 탐색", "🔍 Latent Factors & Unobserved Confounding")}</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="card" style="font-size:0.9em">
      <b>{t("왜 잠재 요인을 봐야 하는가?", "Why look at latent factors?")}</b><br>
      {t("부트스트랩 강도가 낮거나 방향이 불확실한 엣지는 <u>직접 측정되지 않은 제3의 변수</u>(잠재 교란 요인)가 관계를 왜곡할 수 있습니다. 아래에서 LLM이 제안한 잠재 구성개념과 그 proxy 변수를 확인하세요.",
         "For edges with low bootstrap strength or uncertain direction, an <u>unmeasured third variable</u> (a latent confounder) may be distorting the relationship. Review the LLM-proposed latent constructs and their proxy variables below.")}
    </div>
    """, unsafe_allow_html=True)

    sub1, sub2 = st.tabs([t("💜 잠재 구성개념 카드", "💜 Latent construct cards"), t("⚠️ 약한 통계 엣지 분석", "⚠️ Weak statistical edge analysis")])

    # --- Latent constructs ---
    with sub1:
        if latent_edges.empty:
            st.info(t("잠재 구성개념 엣지(wl_edges_latent_final)가 없습니다.", "No latent construct edges (wl_edges_latent_final)."))
        else:
            constructs = latent_edges["from_var"].unique()
            st.caption(t(f"LLM이 제안한 잠재 구성개념 {len(constructs)}개", f"{len(constructs)} LLM-proposed latent constructs"))

            # Gemini proxy 제안 버튼 (캐시 우선)
            proxy_key = f"proxy_api_{dataset}"
            has_key = gemini_report.get_api_key() is not None
            # 실시간 생성분(session) 우선, 없으면 미리 구워 둔 DB 캐시(오프라인 데모용)
            proxy_cache = st.session_state.get(proxy_key) or db.load_latent_proxies(dataset)

            pc1, pc2 = st.columns([1, 2])
            with pc1:
                if proxy_cache:
                    proxy_gen = st.button(t("🔄 proxy 다시 제안", "🔄 Re-suggest proxies"), use_container_width=True,
                                          disabled=not has_key, key="regen_proxy")
                else:
                    proxy_gen = st.button(t("🤖 Gemini로 측정 proxy 변수 제안", "🤖 Suggest measurable proxy variables via Gemini"), type="primary",
                                          use_container_width=True, disabled=not has_key, key="gen_proxy")
            with pc2:
                if proxy_cache:
                    st.caption(t("✅ 저장된 proxy 제안을 표시 중입니다 (API 호출 없음).",
                                 "✅ Showing saved proxy suggestions (no API call)."))
                elif not has_key:
                    st.caption(t("⚠️ GEMINI_API_KEY 미설정 — `.streamlit/secrets.toml` 확인",
                                 "⚠️ GEMINI_API_KEY not set — check `.streamlit/secrets.toml`"))
                else:
                    st.caption(t("기존 변수 대신, 측정 가능한 **새 pre-treatment proxy 변수**를 Gemini가 제안합니다.",
                                 "Instead of existing variables, Gemini suggests **new measurable pre-treatment proxy variables**."))

            if proxy_gen:
                with st.spinner(t("🤖 Gemini API 추론 중... proxy 변수 제안 생성 (최대 ~30초)",
                                  "🤖 Running Gemini API inference... generating proxy suggestions (up to ~30s)")):
                    try:
                        st.session_state[proxy_key] = gemini_report.suggest_latent_proxies(dataset)
                        proxy_cache = st.session_state[proxy_key]
                        st.success(t("✅ proxy 제안 생성 완료", "✅ Proxy suggestions generated"))
                    except RuntimeError as e:
                        st.error(str(e))
                    except Exception as e:
                        st.error(t(f"❌ proxy 제안 실패: {e}", f"❌ Failed to generate proxy suggestions: {e}"))
                        raw = getattr(e, "raw", None)
                        if raw:
                            with st.expander(t("🔎 RAW 응답 보기", "🔎 View RAW response")):
                                st.code(raw)

            by_construct = (proxy_cache or {}).get("by_construct", {})

            for construct in constructs:
                rows_c = latent_edges[latent_edges["from_var"] == construct]
                full_rationale = str(rows_c.iloc[0].get("rationale", "")) if len(rows_c) else ""

                # Gemini가 제안한 새 proxy 변수
                suggested = by_construct.get(construct, [])
                if suggested:
                    proxy_html = ""
                    for p in suggested:
                        timing = str(p.get("timing", "")).lower()
                        is_pre = "pre" in timing
                        badge_cls = "chip-green" if is_pre else "chip-orange"
                        badge_lbl = "pre-treatment ✅" if is_pre else "post-treatment ⚠️"
                        proxy_html += (
                            f'<div style="margin:8px 0;padding:8px 12px;background:#FFFFFF;'
                            f'border-radius:6px;border-left:3px solid '
                            f'{C["accent"] if is_pre else C["warning"]}">'
                            f'<b>{p.get("name","")}</b> '
                            f'<span class="chip {badge_cls}" style="font-size:0.72em">{badge_lbl}</span>'
                            f'<br><span style="font-size:0.85em;color:#555">{t("📏 측정", "📏 Measure")}: {p.get("measurement","")}</span>'
                            f'<br><span style="font-size:0.85em">{p.get("rationale","")}</span>'
                            f'</div>'
                        )
                    proxy_section = f'<b>{t("🤖 제안 측정 proxy 변수 (Gemini):", "🤖 Suggested measurement proxy variables (Gemini):")}</b>{proxy_html}'
                else:
                    proxy_section = (f'<span style="font-size:0.85em;color:#888">'
                                     f'{t("위 버튼으로 측정 가능한 새 proxy 변수를 생성하세요.", "Use the button above to generate new measurable proxy variables.")}</span>')

                st.markdown(f"""
                <div class="card-purple">
                  <b>💜 {construct}</b>
                  <span class="chip chip-purple" style="float:right">{t("잠재 구성개념", "Latent construct")}</span>
                  <br><br>
                  <b>{t("왜 직접 측정이 어렵나요?", "Why is it hard to measure directly?")}</b><br>
                  <span style="font-size:0.88em">{full_rationale}</span>
                  <br><br>
                  {proxy_section}
                </div>
                """, unsafe_allow_html=True)

            # Latent DAG visualization — 누적(stat+wl_edges+latent) + 레이어 토글
            st.markdown(f"**{t('잠재 구성개념 포함 DAG (누적 레이어)', 'DAG including latent constructs (cumulative layers)')}**")
            lc1, lc2, lc3, lc4 = st.columns(4)
            lay_stat = lc1.checkbox(t("통계 엣지(stat)", "Statistical edges (stat)"), value=True, key="lay_stat")
            lay_cand = lc2.checkbox(t("후보 엣지(wl_edges)", "Candidate edges (wl_edges)"), value=True, key="lay_cand")
            lay_lat  = lc3.checkbox(t("잠재 엣지(latent)", "Latent edges (latent)"), value=True, key="lay_lat")
            lay_leg  = lc4.checkbox(t("범례 표시", "Show legend"), value=True, key="lay_leg")
            if lay_leg:
                st.markdown(_legend_html(), unsafe_allow_html=True)
            if not (lay_stat or lay_cand or lay_lat):
                st.info(t("표시할 레이어를 1개 이상 선택하세요.", "Select at least one layer to display."))
            else:
                h_lat = _pyvis_html(
                    stat_edges=stat_edges, candidate_edges=wl_edges, latent_edges=latent_edges,
                    var_meta=var_meta, theory_map=theory_map,
                    rationale_map=rationale_map,
                    show_stat=lay_stat, show_candidate=lay_cand, show_latent=lay_lat,
                    height="480px",
                )
                st.caption(t("레이어를 켜고 끄며 stat·후보(wl_edges)·잠재 엣지를 겹쳐 볼 수 있습니다.",
                             "Toggle layers to overlay stat / candidate (wl_edges) / latent edges."))
                components.html(h_lat, height=490, scrolling=False)

    # --- Weak edges ---
    with sub2:
        weak = get_weak_edges(stat_edges, threshold=0.90)
        uncertain = wl_edges[wl_edges["reason"] == "uncertain_dir"] if not wl_edges.empty else pd.DataFrame()

        if weak.empty and uncertain.empty:
            st.success(t("모든 통계 엣지의 부트스트랩 강도가 0.90 이상입니다.", "All statistical edges have bootstrap strength ≥ 0.90."))
        else:
            if not weak.empty:
                st.markdown(t(f"""
                <div class="card-warn">
                  <b>⚠️ 부트스트랩 강도 낮은 확정 엣지 ({len(weak)}개)</b><br>
                  강도 &lt; 0.90인 엣지는 방향 확신이 약하며, 미관측 교란 요인 가능성이 있습니다.
                </div>
                """,
                f"""
                <div class="card-warn">
                  <b>⚠️ Confirmed edges with low bootstrap strength ({len(weak)})</b><br>
                  Edges with strength &lt; 0.90 have weak directional confidence and may involve unobserved confounders.
                </div>
                """), unsafe_allow_html=True)
                disp_weak = weak.rename(columns={"from_var":t("원인", "Cause"),"to_var":t("결과", "Effect"),
                                                   "strength":t("강도", "Strength"),"tag":t("상태", "Status")})
                st.dataframe(disp_weak, use_container_width=True)

                # Match latent to weak
                matched = match_latent_to_weak(weak, latent_edges)
                if not matched.empty:
                    st.markdown(t(f"""
                    <div class="card-purple">
                      <b>💜 관련 잠재 교란 요인 후보 ({len(matched)}개 엣지)</b><br>
                      아래 잠재 구성개념이 위 약한 엣지에 영향을 줄 수 있습니다.
                      Proxy 변수를 추가 측정하거나 통계 모형에 포함하는 것을 권장합니다.
                    </div>
                    """,
                    f"""
                    <div class="card-purple">
                      <b>💜 Candidate related latent confounders ({len(matched)} edges)</b><br>
                      The latent constructs below may influence the weak edges above.
                      Measuring additional proxy variables or including them in the statistical model is recommended.
                    </div>
                    """), unsafe_allow_html=True)
                    for _, mrow in matched.iterrows():
                        st.markdown(
                            f"- 💜 **{mrow['from_var']}** → `{mrow['to_var']}` "
                            f": {str(mrow.get('rationale',''))[:150]}"
                        )

            if not uncertain.empty:
                with st.expander(t(f"🔸 방향 불확실 후보 엣지 ({len(uncertain)}개) — 미채택 원인 파악",
                                   f"🔸 Direction-uncertain candidate edges ({len(uncertain)}) — why they were not adopted")):
                    disp_unc = uncertain[["from_var","to_var","rationale"]].rename(
                        columns={"from_var":t("원인", "Cause"),"to_var":t("결과", "Effect"),"rationale":t("LLM 판단 근거", "LLM rationale")}
                    )
                    st.dataframe(disp_unc, use_container_width=True, height=220)

        # Proxy variable guide from theory_mapping
        if not theory_map.empty:
            proxy_rows = theory_map[theory_map["proxy_type"].isin(["proxy","direct_measure"])] \
                         if "proxy_type" in theory_map.columns else pd.DataFrame()
            if not proxy_rows.empty:
                with st.expander(t("📋 변수별 Proxy / 측정 역할 전체 보기", "📋 View all proxy / measurement roles per variable")):
                    st.dataframe(proxy_rows, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — 연구자 질의
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown(f'<div class="section-header">{t("💬 연구자 질의 — DB 기반 검색", "💬 Researcher Query — DB-based Search")}</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="card" style="font-size:0.9em">
      <b>{t("사용법:", "How to use:")}</b> {t("변수명, 이론명, 인과 관계 키워드를 입력하면 DB에서 관련 내용을 검색합니다.", "Enter a variable name, theory name, or causal-relation keyword to search the DB for related content.")}<br>
      <span style="color:{C['medium']}">{t("예", "e.g.")}: "Discussion", "motivation", "autonomy", "MSLQ", "rationale"</span><br><br>
      {t("⚠️ 현재 <b>실시간 LLM 추론은 비활성화</b>되어 있습니다 (API 크레딧 부족). DB에 없는 질의는 결과가 반환되지 않습니다.",
         "⚠️ <b>Live LLM inference is currently disabled</b> (insufficient API credit). Queries not present in the DB return no results.")}
    </div>
    """, unsafe_allow_html=True)

    query = st.text_input(
        t("🔍 질의 입력", "🔍 Enter query"),
        placeholder=t("예: Discussion_level 이 Class 에 미치는 영향은?", "e.g. What effect does Discussion_level have on Class?"),
        key="user_query"
    )

    # Quick buttons
    st.markdown(f"**{t('빠른 검색:', 'Quick search:')}**")
    qcols = st.columns(4)
    quick_queries = ["motivation", "engagement", "performance", "autonomy"]
    for i, qc in enumerate(quick_queries):
        if qcols[i].button(qc, key=f"quick_{qc}"):
            query = qc

    if query and len(query.strip()) >= 2:
        with st.spinner(t("DB 검색 중...", "Searching DB...")):
            results = db.search_db(dataset, query.strip())

        if results:
            st.success(t(f"'{query}' 관련 항목 {sum(len(v) for v in results.values())}개 발견",
                         f"Found {sum(len(v) for v in results.values())} items related to '{query}'"))
            for section, df in results.items():
                with st.expander(f"📂 {section} ({len(df)}{t('건', '')})", expanded=True):
                    # Truncate long text columns for display
                    disp = df.copy()
                    for col in disp.columns:
                        if disp[col].dtype == object:
                            disp[col] = disp[col].apply(
                                lambda x: str(x)[:200] + "…" if len(str(x)) > 200 else str(x)
                            )
                    st.dataframe(disp, use_container_width=True)

            # Also show in DAG if variable names found
            found_vars = []
            for _, df in results.items():
                for col in ["var_name", "from_var", "to_var"]:
                    if col in df.columns:
                        found_vars.extend(df[col].tolist())
            found_vars = [v for v in set(found_vars) if v in all_vars]

            if found_vars:
                st.markdown(f"**{t('관련 변수를 DAG에서 하이라이트:', 'Highlight related variables in the DAG:')}**")
                h_q = _pyvis_html(
                    stat_edges=stat_edges,
                    var_meta=var_meta, theory_map=theory_map,
                    rationale_map=rationale_map,
                    show_stat=True,
                    highlight_nodes=set(found_vars),
                    height="420px",
                )
                components.html(h_q, height=430, scrolling=False)

        else:
            st.markdown(t(f"""
            <div class="card-warn">
              <b>🔍 '{query}'에 대한 검색 결과가 없습니다.</b><br><br>
              현재 저장된 DB 내용에서 일치하는 항목을 찾지 못했습니다.<br>
              실시간 LLM 추론(Gemini API)은 현재 <b>비활성화</b> 상태입니다.<br><br>
              💡 <b>대안:</b> Colab에서 추가 해석을 생성하여 <code>dag_interpretations</code>에 저장 후 재검색해보세요.
            </div>
            """,
            f"""
            <div class="card-warn">
              <b>🔍 No search results for '{query}'.</b><br><br>
              No matching items were found in the currently stored DB content.<br>
              Live LLM inference (Gemini API) is currently <b>disabled</b>.<br><br>
              💡 <b>Alternative:</b> Generate additional interpretations in Colab, save them to <code>dag_interpretations</code>, and search again.
            </div>
            """), unsafe_allow_html=True)

    # Static FAQ from DB
    st.markdown("---")
    st.markdown(f"**{t('📘 DB 기반 주요 인사이트', '📘 Key DB-based insights')}**")
    faq_cols = st.columns(2)

    with faq_cols[0]:
        if not stat_edges.empty:
            top = stat_edges.nlargest(3, "strength") if "strength" in stat_edges.columns else stat_edges.head(3)
            items = [f"**{r['from_var']}** → **{r['to_var']}** ({t('강도', 'strength')}: {r.get('strength',1.0):.2f})"
                     for _, r in top.iterrows()]
            st.markdown(f"""
            <div class="card">
              <b>{t("🏆 가장 강한 인과 관계 Top 3", "🏆 Top 3 strongest causal relations")}</b><br>
              {"<br>".join(f"• {i}" for i in items)}
            </div>
            """, unsafe_allow_html=True)

    with faq_cols[1]:
        if not latent_edges.empty:
            constructs = latent_edges["from_var"].unique()[:3]
            items = [f"**{c}** → {', '.join(latent_edges[latent_edges['from_var']==c]['to_var'].tolist())}"
                     for c in constructs]
            st.markdown(f"""
            <div class="card-purple">
              <b>{t("💜 주요 잠재 구성개념 (직접 측정 불가)", "💜 Key latent constructs (not directly measurable)")}</b><br>
              {"<br>".join(f"• {i}" for i in items)}
            </div>
            """, unsafe_allow_html=True)
