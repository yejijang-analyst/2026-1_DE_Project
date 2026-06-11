# -*- coding: utf-8 -*-
"""게임 데이터셋용 임시 인과 해석을 temp_report 테이블에 적재."""
import sqlite3, os

BASE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE, "game_pipeline.db")

BRIEF = (
    "LV는 DIV, RT, PG, COOR, HOV, NAV 등 다수의 행동 변수와 연결된 중심 노드로 확인되었다. "
    "해당 노드는 다수의 하위 행동 변수에 대해 공통 원인으로 기능하는 구조를 형성하며, 이로 인해 LV를 "
    "조건화하지 않을 경우 하위 변수 간 비인과적 연관성이 발생할 수 있는 교란 변수로서의 역할이 확인된다. "
    "또한 LV는 PERF에 직접 연결되는 동시에 RT를 경유하는 간접 경로(LV→RT→PERF)도 함께 형성하여, 과업 "
    "구조가 성취도에 직접 효과와 행동 매개 경로를 동시에 갖는 복합적 구조가 지지된다. PERF는 LV 및 RT를 "
    "직접 부모 노드로 가지는 구조가 확인되었다. LV→PERF의 직접 경로와 LV→RT→PERF의 간접 경로가 공존하는 "
    "구조에서 RT는 부분 매개 변수로 기능하는 것으로 확인된다. LV와 RT가 PERF에 수렴하는 구조는 충돌부"
    "(Collider)를 형성하며, PERF를 조건화할 경우 LV와 RT 간 비인과적 연관성이 유도될 수 있어 해석 시 "
    "주의가 요구된다. RT는 LV, DIV, HOV를 부모 노드로 가지며 PERF 및 TIME에 연결되는 구조가 확인되었다. "
    "HOV→RT→PERF의 간접 경로가 형성되어 HOV가 RT를 매개로 PERF에 간접 연결되는 구조가 지지된다. "
    "LV, DIV, HOV가 RT에 수렴하는 구조에서 RT는 충돌부에 해당하며, RT 조건화 시 해당 부모 노드 간 "
    "비인과적 연관성이 유도될 수 있다. HOV는 LV 및 DIV를 부모 노드로 가지며 RT, NAV, TIME에 연결되는 "
    "구조가 확인된다. DIV는 LV, FS, PG를 부모 노드로 가지며 RT, HOV, COOR에 연결되는 구조가 확인되었다. "
    "LV, FS, PG가 DIV에 수렴하는 구조는 충돌부에 해당하여, DIV 조건화 시 해당 부모 노드 간 비인과적 "
    "연관성 유도 가능성이 확인된다. FS는 DIV에만 연결되는 외생 변수로 확인된다. NAV는 HOV 및 LV를 부모 "
    "노드로 가지며 COOR에 연결되는 구조가 확인되었다. COOR은 LV, DIV, NAV, PG를 부모 노드로 가지는 단말 "
    "노드로 확인된다. PG는 DIV 및 COOR에 연결되는 구조가 확인된다. TIME은 LV, RT, HOV를 부모 노드로 "
    "가지는 단말 노드로 확인된다."
)

RICH = (
    "본 연구에서 도출된 인과 구조 분석 결과는 게이미피케이션의 핵심 이론인 몰입 이론과 자기결정성 이론이 "
    "실제 로그 데이터 환경에서 어떠한 인과적 기제로 작동하는지를 해석할 수 있는 근거를 제공한다. 이론적 "
    "심리 구성 개념과 로그 기반 행동 지표를 연결하여 그 작동 메커니즘을 분석하였으며, 개별 변수의 영향력 "
    "검토를 넘어 게이미피케이션 요소가 학습자의 심리 상태를 반영한 행동 지표를 매개로 학습 성과로 이어지는 "
    "인과 경로를 탐색적으로 제시하였다. 구체적인 논의와 의의는 다음과 같다. 첫째, HOV→RT→PERF의 간접 경로 "
    "구조는 몰입 이론의 주의 집중 및 행동 지속성 개념과 대응하는 결과로 해석된다. 몰입 이론에서는 학습자가 "
    "과제에 완전히 몰입할 때 주의가 특정 대상에 집중되고 이것이 학습 성과로 이어진다고 설명한다"
    "(Csikszentmihalyi, 1990). HOV가 RT를 매개로 PERF에 간접 연결되는 구조는 주의 집중 관련 행동이 "
    "상호작용 행동을 경유하여 성취도와 연결되는 경향을 반영한다. 아울러 과업 난이도와 단계가 주의 집중 및 "
    "활동 탐색 행동을 경유하여 학습 성과와 연결되는 다단계 간접 경로 구조는, 몰입 이론의 도전-기술 균형 "
    "개념과 맥락을 같이하는 구조적 경향으로 해석될 수 있다. 부가적으로, TIME이 LV, RT, HOV를 부모 노드로 "
    "가지는 단말 구조로 확인되어 세션 소요 시간이 맥락 및 상호작용·집중 행동의 결과 변수로 기능하는 경향이 "
    "관찰되었다. 이는 학습 시간 자체보다 시간 내에 나타나는 행동 패턴이 성취도와 더 직접적으로 관련될 수 "
    "있음을 시사한다. 둘째, DIV→RT→PERF의 간접 경로 구조는 자기결정성 이론의 자율성 개념과 대응하는 "
    "결과로 해석된다. 자기결정성 이론에서는 학습자가 스스로 행동을 선택하고 조절할 때 내재적 동기가 "
    "강화된다고 설명한다(Ryan & Deci, 2000). DIV가 RT를 매개로 PERF에 간접 연결되는 구조는 자율적 탐색 "
    "행동이 상호작용 행동을 경유하여 성취도와 연결되는 경향을 반영한다. 아울러 FS가 DIV에만 연결되는 외생 "
    "변수로 확인되어, 초기 학습 진입 행동이 활동 다양성의 선행 조건으로 기능하는 구조가 관찰되었다. 이는 "
    "게이미피케이션 환경에서 학습 참여 시작 시점의 자발성이 이후 자율적 탐색 행동의 범위와 관련될 수 "
    "있음을 시사한다. 셋째, RT가 LV, DIV, HOV를 부모 노드로 가지며 PERF에 직접 연결되는 구조는 "
    "자기결정성 이론의 관계성 개념과 대응하는 결과로 해석된다. 다만 본 연구의 RT는 학습 환경 내 사람 "
    "캐릭터와의 관계적 상호작용 빈도로 조작화된 것으로, 대인 간 사회적 연결감이라는 관계성의 원래 개념과 "
    "완전히 부합하지 않는 조작적 근사값임을 고려하여 해석할 필요가 있다. 이러한 한계를 전제하더라도, 조작적 "
    "근사 수준에서 RT가 PERF에 대한 유일한 직접 행동 지표로 기능하는 구조는 게이미피케이션 환경에서 "
    "상호작용 행동이 학습 성과와 가장 근접하게 연결되는 행동 변수일 수 있음을 시사한다."
)

COLLIDER_WARN = (
    "PERF, RT, DIV가 충돌부(Collider)로 식별되었다. "
    "PERF 조건화 시 LV와 RT 간, RT 조건화 시 LV·DIV·HOV 간, DIV 조건화 시 LV·FS·PG 간 "
    "비인과적 연관성이 유도될 수 있으므로, 해당 노드를 통제 변수로 포함할 때 주의가 필요하다."
)

MEDIATOR_NOTE = (
    "RT는 LV→RT→PERF, HOV→RT→PERF, DIV→RT→PERF 경로의 핵심 매개 변수로 기능한다. "
    "LV→PERF의 순수 직접 효과를 측정하려는 경우 RT를 통제하면 매개 경로가 차단되어 "
    "과도 통제(over-control) 편향이 발생하므로, 직접 효과 추정 시 RT를 조건화하지 않아야 한다."
)

PROXY_GUIDE = (
    "[관계성(Relatedness)]: RT — 학습 환경 내 NPC 등 사람 캐릭터와의 관계적 상호작용 빈도로 "
    "조작화된 근사값. 대인 간 사회적 연결감 자체는 직접 측정 불가하므로 행동 빈도로 대리한다. / "
    "[자율성(Autonomy)]: DIV·NAV — 자율적 탐색·선택 행동의 다양성으로 대리 측정. / "
    "[주의 집중(Attention/Flow)]: HOV — 몰입 상태의 주의 집중 정도를 행동 로그로 대리한다."
)

conn = sqlite3.connect(DB)
conn.execute("""
    CREATE TABLE IF NOT EXISTS temp_report (
        dataset        TEXT PRIMARY KEY,
        brief          TEXT,
        rich           TEXT,
        collider_warn  TEXT,
        mediator_note  TEXT,
        proxy_guide    TEXT,
        created_at     TEXT DEFAULT (datetime('now','localtime'))
    )
""")
conn.execute("""
    INSERT INTO temp_report (dataset, brief, rich, collider_warn, mediator_note, proxy_guide)
    VALUES (?, ?, ?, ?, ?, ?)
    ON CONFLICT(dataset) DO UPDATE SET
        brief         = excluded.brief,
        rich          = excluded.rich,
        collider_warn = excluded.collider_warn,
        mediator_note = excluded.mediator_note,
        proxy_guide   = excluded.proxy_guide,
        created_at    = datetime('now','localtime')
""", ("game", BRIEF, RICH, COLLIDER_WARN, MEDIATOR_NOTE, PROXY_GUIDE))
conn.commit()

row = conn.execute("SELECT dataset, length(brief), length(rich), created_at FROM temp_report").fetchall()
conn.close()
print("temp_report 적재 완료:", row)
