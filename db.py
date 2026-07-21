import sqlite3
import os
import json
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))

DB_PATHS = {
    "xapi":   os.path.join(BASE, "xapi_pipeline.db"),
    "game":   os.path.join(BASE, "game_pipeline.db"),
    "assist": os.path.join(BASE, "assist_pipeline.db"),
}

DATASET_LABELS = {
    "xapi":   "🎓 xAPI 교실 학습",
    "game":   "🎮 게임 기반 학습",
    "assist": "🤖 ASSISTments 지능형 튜터",
}


def _conn(dataset: str) -> sqlite3.Connection:
    return sqlite3.connect(DB_PATHS[dataset])


def _load(dataset: str, table: str) -> pd.DataFrame:
    try:
        conn = _conn(dataset)
        df = pd.read_sql(f"SELECT * FROM {table}", conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


def load_domain_config(dataset: str) -> dict:
    df = _load(dataset, "domain_config")
    if df.empty:
        return {}
    return dict(zip(df["key"], df["value"]))


def load_variable_meta(dataset: str) -> pd.DataFrame:
    return _load(dataset, "variable_meta")


def load_theory_mapping(dataset: str) -> pd.DataFrame:
    return _load(dataset, "theory_mapping")


def load_theory_meta(dataset: str) -> pd.DataFrame:
    return _load(dataset, "theory_meta")


def load_wl_edges(dataset: str) -> pd.DataFrame:
    return _load(dataset, "wl_edges")


def load_wl_edges_stat(dataset: str) -> pd.DataFrame:
    return _load(dataset, "wl_edges_stat")


def load_wl_edges_stat_raw(dataset: str) -> pd.DataFrame:
    """부트스트랩 결과(블랙리스트 필터 이전) — 통계 DAG로 표시되는 원본."""
    return _load(dataset, "wl_edges_stat_raw")


def load_wl_edges_latent(dataset: str) -> pd.DataFrame:
    return _load(dataset, "wl_edges_latent")


def load_wl_edges_latent_final(dataset: str) -> pd.DataFrame:
    return _load(dataset, "wl_edges_latent_final")


def load_edge_rationales(dataset: str) -> dict:
    """(from_var, to_var) -> LLM rationale 문자열.

    wl_edges, wl_edges_latent_final, wl_edges_latent 에서 rationale 칼럼을 모아
    엣지 툴팁(`llm: ...`)에 사용한다. 먼저 채워진 값이 우선한다.
    """
    out = {}
    for table in ("wl_edges", "wl_edges_latent_final", "wl_edges_latent"):
        df = _load(dataset, table)
        if df.empty or "rationale" not in df.columns:
            continue
        for _, r in df.iterrows():
            key = (r["from_var"], r["to_var"])
            text = str(r.get("rationale", "")).strip()
            if key not in out and text and text.lower() != "nan":
                out[key] = text
    return out


def load_bl_edges(dataset: str) -> pd.DataFrame:
    return _load(dataset, "bl_edges")


def load_interpretation(dataset: str) -> dict:
    """정식 저장 테이블 dag_interpretations 만 읽는다.

    temp_report(임시 적재본)는 표시 대상이 아니므로 폴백하지 않는다.
    실시간 해석은 Gemini API(gemini_report)로 생성한다.
    """
    try:
        conn = _conn(dataset)
        df = pd.read_sql(
            "SELECT * FROM dag_interpretations WHERE dataset = ?",
            conn, params=(dataset,)
        )
        conn.close()
        if not df.empty:
            result = df.iloc[0].to_dict()
            result["_source"] = "dag_interpretations"
            return result
    except Exception:
        pass
    return {}


def load_latent_proxies(dataset: str) -> dict:
    """미리 생성해 저장한 잠재 구성개념별 proxy 제안을 읽는다.

    데모(오프라인) 모드에서 Gemini API 없이도 Tab 4 proxy 카드를 채우기 위한
    캐시 테이블 latent_proxy_suggestions 를 조회한다. 없으면 빈 dict.

    Returns
    -------
    {"_source": "db", "_model": <str>, "by_construct": {construct: [proxy,...]}}
        또는 저장본이 없으면 {}.
    """
    try:
        conn = _conn(dataset)
        df = pd.read_sql(
            "SELECT * FROM latent_proxy_suggestions WHERE dataset = ?",
            conn, params=(dataset,)
        )
        conn.close()
        if df.empty:
            return {}
        row = df.iloc[0].to_dict()
        by_construct = json.loads(row.get("by_construct") or "{}")
        by_construct_en = json.loads(row.get("by_construct_en") or "{}") \
            if row.get("by_construct_en") else {}
        return {
            "_source": "db",
            "_model": row.get("model", ""),
            "by_construct": by_construct,
            "by_construct_en": by_construct_en,
        }
    except Exception:
        return {}


def get_all_vars(dataset: str) -> list:
    edges = load_wl_edges_stat(dataset)
    if edges.empty:
        edges = load_wl_edges(dataset)
    nodes = set()
    if not edges.empty:
        nodes.update(edges["from_var"].tolist())
        nodes.update(edges["to_var"].tolist())
    return sorted(nodes)


def search_db(dataset: str, query: str) -> dict:
    """Keyword search across rationale/description columns in all tables."""
    q = query.lower()
    results = {}

    tables = {
        "변수 정의":   load_variable_meta(dataset),
        "인과 엣지":   load_wl_edges(dataset),
        "이론 매핑":   load_theory_mapping(dataset),
        "잠재 요인":   load_wl_edges_latent(dataset),
    }
    for label, df in tables.items():
        if df.empty:
            continue
        mask = df.apply(lambda row: q in " ".join(str(v) for v in row).lower(), axis=1)
        if mask.any():
            results[label] = df[mask]

    return results
