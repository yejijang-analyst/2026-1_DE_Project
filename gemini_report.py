"""
Gemini API 기반 인과 해석 리포트 생성 (Tab 2).

API 키는 코드에 하드코딩하지 않는다. 다음 우선순위로 조회한다.
  1) 환경변수  GEMINI_API_KEY
  2) Streamlit secrets  (.streamlit/secrets.toml 의 GEMINI_API_KEY)
GitHub에는 .streamlit/secrets.toml 을 절대 커밋하지 않는다(.gitignore 처리).
"""
import os
import json

import db

MODEL_ID = "gemini-2.5-flash-lite"


# ── API 키 ───────────────────────────────────────────────────────────────────
def get_api_key() -> str | None:
    """환경변수 → Streamlit secrets 순으로 키를 찾는다. 없으면 None."""
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key.strip()
    try:
        import streamlit as st
        if "GEMINI_API_KEY" in st.secrets:
            return str(st.secrets["GEMINI_API_KEY"]).strip()
    except Exception:
        pass
    return None


# ── 프롬프트 ──────────────────────────────────────────────────────────────────
def build_prompt(dataset_name, domain, var_meta, theory_map,
                 stat_edges, latent_edges, wl_edges) -> str:
    data_context   = domain.get("data_context",   "")
    theory_context = domain.get("theory_context", "")
    learning_env   = domain.get("learning_env",   "")
    learner_type   = domain.get("learner_type",   "")

    var_desc = dict(zip(var_meta["var_name"], var_meta["description"])) \
               if not var_meta.empty else {}
    var_str  = "\n".join([f"  - {k}: {v}" for k, v in var_desc.items()])

    theory_rows = theory_map.to_dict("records") if not theory_map.empty else []
    theory_str  = "\n".join([
        f"  - {r['var_name']} → [{r['proxy_type']}] {r['construct']} ({r['theory']}): {r['rationale']}"
        for r in theory_rows
    ])

    stat_str = stat_edges[["from_var", "to_var", "strength", "tag"]].to_string(index=False) \
               if not stat_edges.empty else "없음"

    wl_cols  = ["from_var", "to_var", "reason", "rationale"] \
               if "rationale" in wl_edges.columns else ["from_var", "to_var", "reason"]
    wl_str   = wl_edges[wl_cols].to_string(index=False) if not wl_edges.empty else "없음"

    lat_cols   = ["from_var", "to_var", "type", "rationale"] \
                 if "rationale" in latent_edges.columns else ["from_var", "to_var", "type"]
    latent_str = latent_edges[lat_cols].to_string(index=False) if not latent_edges.empty else "없음"

    return f"""
당신은 교육공학, 교육심리학, 인과추론(Causal Inference) 분야의 세계적인 석학이자 전문 연구자입니다.
제공된 데이터셋 맥락, 이론 프레임워크, 그리고 주어진 DAG(방향성 비순환 그래프) 구조적 엣지 정보를 종합하여 학술 논문 수준의 인과 해석 리포트를 작성하세요.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## 1. 데이터셋 맥락 [{dataset_name.upper()}]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
학습 환경: {learning_env} | 학습자 유형: {learner_type}

{data_context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## 2. 이론 프레임워크
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{theory_context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## 3. 변수 정의
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{var_str}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## 4. 변수-이론 구성개념 매핑
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{theory_str}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## 5. 통계 기반 후보 인과 엣지 (wl_edges)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{wl_str}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## 6. 최종 확정 인과 엣지 (wl_edges_stat)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{stat_str}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## 7. 잠재 구성개념 엣지 (wl_edges_latent_final)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{latent_str}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## 🚨 작성 지침 (페르소나 및 스타일)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. 문체: 매우 전문적이고 격식 있는 학술 연구 논문 및 계량 통계학 저널 스타일의 한국어를 구사하세요. (~다 체 사용)

2. 용어의 엄밀성: 인과 구조적 관점에서 '외생 변수', '공통 교란 변수', '매개 변수', '충돌부(Collider)', '조건화(Conditioning)', '비인과적 연관성(Spurious Association)', '단말 노드(Sink Node)' 등의 통계학 및 인과추론 전문 용어를 실제 그래프 구조에 맞춰 정확하게 선택하여 서술해야 합니다.

3. 구체성 및 타 데이터셋 예시 혼용 금지:
   - 대포괄적인 에세이식 요약이나 미사여구를 '절대' 지양하고, 현재 제공된 데이터셋에 존재하는 실제 변수명만을 사용하세요.
   - 가이드라인 예시에 등장하는 타 데이터셋의 변수명을 현재 데이터셋 분석에 절대 혼용하거나 교차 오염시키지 마십시오.

4. 구조 중심의 분석 전개 기법 (가장 중요):
   - 【 brief 】 작성 시 [노드 단위 분석]: 거시 요약 후, 주요 변수들을 개별 문단 헤드라인으로 분리하여 기술하세요. 각 변수가 '어떤 부모 노드'를 가지고 '어떤 자식 노드'로 연결되는지 경로(→)를 명시하고, 해당 노드의 인과 구조적 역할(교란, 매개, 충돌부 등)과 조건화(Conditioning) 시의 통계적 왜곡 위험성을 기계적일 정도로 엄밀하게 분석해야 합니다.
   - 【 rich 】 작성 시 [인과 체인 분석]: 변수를 1:1 화살표 단위로 쪼개어 단순 나열하는 방식을 절대 금지합니다. 대신 최종 결과 변수(Outcome)에 도달하는 '선행 원인 → 매개 변수 → 최종 결과 변수' 형태의 3단계 이상 '다단계 인과 체인(Causal Chain)'을 핵심 줄기로 설정하고, 이 줄기들을 이론적 프레임워크와 결합하여 서술형으로 깊이 있게 논증하세요.

5. 최종 결과 변수(Outcome Node) 중심의 필터링:
   - 제공된 엣지 정보에서 모든 화살표가 최종 수렴하고 더 이상 나가지 않는 '최종 결과 변수(Outcome/종속변수)'를 스스로 식별하세요.
   - 모든 분석 리포트의 결론은 이 최종 결과 변수가 어떻게 인과적으로 형성되는가, 그리고 이 최종 결과 변수가 구조 내에서 '단말 노드(Sink Node)이자 충돌부(Collider)'로서 어떠한 통계적 제약(조건화 시 충돌부 편향 위험)을 가지는가로 귀결되어야 합니다.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## 📋 출력 포맷 및 요구사항
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
반드시 아래 JSON 형식으로만 응답하세요. 마크다운 코드블록(```json)이나 앞뒤 설명 텍스트를 절대로 붙이지 말고, 중괄호 '{{' 로 시작해서 '}}' 로 끝나는 순수 JSON 문자열만 반환해야 합니다.

🚨 [주의] JSON 파싱 에러 방지를 위해, 텍스트 내부에서 문단을 나눌 때 절대로 키보드 Enter(실제 줄바꿈)를 치지 마세요. 문단 구분이 필요한 모든 곳에는 오직 제어 문자 표현인 '\\n\\n'을 문자 그대로 텍스트 안에 삽입하여 한 줄의 연속된 문자열로 출력해야 합니다.

{{
  "brief": "학습된 방향성 비순환 그래프(DAG) 구조의 확률적 의존 관계에 대한 엄밀한 구조적 해석이다. 하이픈, 불릿 기호, 대괄호 등의 특수 서식을 절대 사용하지 말고 오직 순수한 줄글 문단으로만 기술해야 한다. 첫 번째 문단은 '학습된 방향성 비순환 그래프 구조는 학습 행동 변수와 최종 결과 변수 간의 확률적 의존 관계를 나타낸다.'로 시작하여 전체 그래프 구조의 거시적 특징인 핵심 공통 교란 변수 및 종착 매개 경로의 경향성을 종합 요약하여 서술해야 한다. 두 번째 문단부터는 분석 대상 구조에서 식별된 핵심 원인 변수 및 주요 매개 변수들을 개별 문단의 첫 단어로 명시하며 문단을 나누어 기술하되, 해당 노드의 부모와 자식 노드 간의 경로를 명시하고 구조적 역할인 외생 변수, 공통 교란 변수, 매개 변수, 충돌부 중 하나를 규정하며 조건화할 때 유도되는 통계적 왜곡 및 비인과적 연관성 위험성을 엄밀히 분석해야 한다. 마지막 문단은 모든 경로가 최종 수렴하는 최종 결과 변수를 주인공으로 삼아 해당 노드가 단말 노드이자 핵심 충돌부임을 명시하고, 이를 조건화하여 표본을 통제할 때 부모 노드들 간에 어떠한 충돌부 편향과 비인과적 가짜 연관성이 유도되는지 해석상의 주의점을 반드시 강력한 경고 문장으로 서술해야 한다.",
  "rich": "제시된 이론적 프레임워크와 도출된 인과 구조를 유기적으로 결합한 서술형 학술 논증이다. 변수를 하나씩 쪼개어 단순 나열하는 기호나 서식을 절대 지양하고, 모든 문단이 최종 결과 변수(Outcome)의 형성과 성과 발현 메커니즘이라는 하나의 결론으로 수렴하도록 총 4개의 순수 줄글 문단으로만 전개해야 한다. 첫 번째 문단은 구조 내에서 관찰되는 가장 뚜렷한 다단계 간접 경로 체인을 선택하여 이를 서두에 문장 형태로 제시하고, 해당 경로가 이론적 프레임워크의 어떤 인지적, 정서적, 행동적 메커니즘을 반영하는지 관련 학술 연구 및 저자 연도 인용구와 매핑하여 인과 기제 중심으로 서술하되 이 다단계 흐름이 궁극적으로 최종 결과 변수의 긍정적 혹은 부정적 성과를 어떻게 결정짓는지 종착지 중심으로 결론을 맺어야 한다. 두 번째 문단은 첫 번째와 확연히 구분되는 또 다른 매개 경로 체인을 선택하여 서두에 배치하고, 선행 요인이나 시스템 환경적 특성이 어떠한 행동적 또는 심리적 매개 변수들을 경유하여 최종 결과 변수로 전이되는지 그 구조적 인과 흐름의 교육학적 의미를 규명하며 이 경로 역시 최종 결과 변수에 도달하는 총 효과의 관점에서 문단을 마무리해야 한다. 세 번째 문단은 데이터셋의 미관측 잠재 구성개념들이 직접 측정 불가능한 이론적 이유를 밝히고, 이 잠재적 특성들이 본 데이터셋에 존재하는 구체적인 관측 행동 프록시 변수들의 어떠한 결합 패턴을 통해 인과적으로 간접 식별 및 추론되고 있는지 설명하되 이러한 잠재적 전제 조건들이 결국 최종 결과 변수의 발현을 어떻게 인과적으로 추동하고 지지하는지 연결하여 결론을 도출해야 한다. 네 번째 문단은 이 모든 인과 구조적 발견이 실제 현장 운영이나 시스템 대시보드 구축 등 최종 결과 변수를 최적화하고 향상시키기 위해 어떠한 실천적 개입 전략을 제공하는지 제언하고, 이와 함께 통제되지 못한 미관측 교란 요인으로 인한 백도어 경로의 개방 가능성 및 최종 결과 변수 자체를 조건화할 때 발생하는 충돌부 편향이 전체 인과 네트워크 해석에 어떤 치명적인 왜곡을 초래하는지 그 통계학적 한계점을 엄밀한 마무리 문장으로 제시해야 한다."
}}
"""


# ── 생성 ──────────────────────────────────────────────────────────────────────
def generate_interpretation(dataset: str, model_id: str = MODEL_ID) -> dict:
    """Gemini API를 호출해 {brief, rich} 해석을 생성한다.

    Raises
    ------
    RuntimeError      : API 키 미설정
    json.JSONDecodeError : 응답이 JSON으로 파싱되지 않음(.raw 로 원문 확인)
    """
    key = get_api_key()
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY가 설정되지 않았습니다. "
            ".streamlit/secrets.toml 또는 환경변수 GEMINI_API_KEY를 확인하세요."
        )

    from google import genai  # 지연 임포트 — 앱 시작 속도 보호

    domain     = db.load_domain_config(dataset)
    var_meta   = db.load_variable_meta(dataset)
    theory_map = db.load_theory_mapping(dataset)
    stat_df    = db.load_wl_edges_stat(dataset)
    latent_df  = db.load_wl_edges_latent_final(dataset)
    wl_df      = db.load_wl_edges(dataset)

    prompt = build_prompt(dataset, domain, var_meta, theory_map,
                          stat_df, latent_df, wl_df)

    client   = genai.Client(api_key=key)
    response = client.models.generate_content(model=model_id, contents=prompt)
    raw      = (response.text or "").strip()
    clean    = raw.replace("```json", "").replace("```", "").strip()

    try:
        parsed = json.loads(clean)
    except json.JSONDecodeError as e:
        e.raw = raw  # type: ignore[attr-defined]
        raise

    parsed["_source"] = "gemini_api"
    parsed["_model"]  = model_id
    return parsed


# ── 잠재 구성개념 → 측정 proxy 변수 제안 ──────────────────────────────────────
def build_proxy_prompt(dataset_name, domain, var_meta, outcome_vars, constructs) -> str:
    """잠재 구성개념별로 '새로운' 관측 proxy 변수를 제안하도록 하는 프롬프트."""
    learning_env = domain.get("learning_env", "")
    learner_type = domain.get("learner_type", "")
    data_context = domain.get("data_context", "")

    existing = ", ".join(var_meta["var_name"].tolist()) if not var_meta.empty else "없음"
    outcome_str = ", ".join(outcome_vars) if outcome_vars else "(명시되지 않음)"
    construct_block = "\n".join(
        f"  - {c['construct']}: {c['rationale']}" for c in constructs
    )

    return f"""
당신은 교육심리 측정(Measurement)과 인과추론(Causal Inference) 전문가입니다.
아래 학습 맥락에서 각 '잠재 구성개념(latent construct)'을 실제로 측정할 수 있는
'새로운 관측 proxy 변수'를 제안하세요.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## 학습 맥락 [{dataset_name.upper()}]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
학습 환경: {learning_env} | 학습자 유형: {learner_type}
{data_context}

## 최종 결과 변수(Outcome)
{outcome_str}
→ 이 결과 변수보다 '시간적으로 이전(pre-treatment)' 또는 그와 독립적으로 관측 가능한
   proxy를 우선 제안하세요. 결과 변수 '이후'에 발생/측정되는 post-treatment 변수는 배제합니다.

## 이미 데이터셋에 존재하는 변수 (그대로 재언급 금지)
{existing}

## proxy를 제안할 잠재 구성개념
{construct_block}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## 작성 지침
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. 위 '이미 존재하는 변수'를 그대로 쓰지 말고, 구체적이고 측정 가능한 '새로운' 지표를 제안하세요.
   (예: Task Value → '과제 제출률', '과제 평균 점수', '자발적 심화학습 시간')
2. 가능한 한 pre-treatment confounder(처치·결과 이전 또는 그와 독립으로 관측되는 변수)를 제안하고,
   outcome 이후에 발생하는 post-treatment 변수는 최대한 배제하세요.
3. 각 proxy마다 timing 을 'pre-treatment' 또는 'post-treatment' 중 하나로 명시하세요(되도록 pre-treatment).
4. 각 구성개념당 2~3개의 proxy를 제안하세요. 한국어로 작성하세요.

반드시 아래 JSON 형식만 반환하세요. 마크다운 코드블록이나 설명 텍스트를 절대 붙이지 말고,
'{{'로 시작해 '}}'로 끝나는 순수 JSON 문자열만 출력하세요.

{{
  "constructs": [
    {{
      "construct": "구성개념 이름(입력과 동일하게)",
      "proxies": [
        {{
          "name": "과제 제출률",
          "measurement": "학기 중 마감 기한 내 제출한 과제 비율(%)",
          "timing": "pre-treatment",
          "rationale": "이 지표가 해당 잠재 구성개념을 어떻게 대리하는지 1~2문장"
        }}
      ]
    }}
  ]
}}
"""


def suggest_latent_proxies(dataset: str, model_id: str = MODEL_ID) -> dict:
    """잠재 구성개념별 새 proxy 변수를 Gemini로 추론. {construct: [proxy,...]} 반환."""
    key = get_api_key()
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY가 설정되지 않았습니다. "
            ".streamlit/secrets.toml 또는 환경변수 GEMINI_API_KEY를 확인하세요."
        )

    from google import genai

    domain     = db.load_domain_config(dataset)
    var_meta   = db.load_variable_meta(dataset)
    theory_map = db.load_theory_mapping(dataset)
    latent_df  = db.load_wl_edges_latent_final(dataset)

    if latent_df.empty:
        return {"_source": "gemini_api", "_model": model_id, "by_construct": {}}

    outcome_vars = (
        theory_map.loc[theory_map["proxy_type"] == "outcome", "var_name"].tolist()
        if not theory_map.empty and "proxy_type" in theory_map.columns else []
    )
    # 구성개념별 대표 rationale 1개
    constructs = []
    for c in latent_df["from_var"].unique():
        row = latent_df[latent_df["from_var"] == c].iloc[0]
        constructs.append({"construct": c, "rationale": str(row.get("rationale", ""))})

    prompt   = build_proxy_prompt(dataset, domain, var_meta, outcome_vars, constructs)
    client   = genai.Client(api_key=key)
    response = client.models.generate_content(model=model_id, contents=prompt)
    raw      = (response.text or "").strip()
    clean    = raw.replace("```json", "").replace("```", "").strip()

    try:
        parsed = json.loads(clean)
    except json.JSONDecodeError as e:
        e.raw = raw  # type: ignore[attr-defined]
        raise

    by_construct = {c["construct"]: c.get("proxies", []) for c in parsed.get("constructs", [])}
    return {"_source": "gemini_api", "_model": model_id, "by_construct": by_construct}
