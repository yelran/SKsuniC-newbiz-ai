"""
F3-2. 조직계열 4개 항목 점수 계산 (확정 배점표 2026-08-06 기준)

■ 배점 (합 55점 — 시장계열 45점과 합쳐 100점)
  1. 조직 역량 적합도 20 = A입력유형적합 7 + A수행작업적합 6 + B특허분류매칭 4 + B로드맵연계 3
  2. 역량 전이 가능성 15 = A범용역량전이 6 + A입력유형전이 6 + B유휴특허 3
  3. 부족 역량 수준   15 = A미매칭역량 5 + B특허미커버 3 + C도메인인력부재 7
  4. 실행 가능성       5 = C도메인전문성 5


■ 데이터 축 변경 (중요)
  이전: DB 태그 → alias_map → F1 capability 이름(11개)
  현재: DB 태그 → 역량어휘_매핑.csv(177행) → 표준역량ID(32개) → 표준역량_정의.csv
  이유: 확정 배점표의 세부항목이 표준역량_정의.csv의 컬럼과 1:1로 대응한다.
        역량성격(입력유형/수행작업/도메인/기반) · 특허분류 · 특허건수 ·
        로드맵연계 · 전담인력 · 전이출처 · 조직보유수준(★0~5)
        F1 JSON에는 이 정보가 없어서 세부항목을 계산할 수 없다.

■ 계산 원칙 3가지
  ① 각 세부항목은 '요구역량 중 조직이 얼마나 커버하는가'를 0~1 비율로 구한 뒤 배점을 곱한다.
  ② 유무(0/1)가 아니라 조직보유수준(★1~5)을 가중치로 쓴다. ★5 역량이 매칭되면 ★2보다 높다.
  ③ 평가 대상이 하나도 없으면 0점이 아니라 None을 반환한다(F2-5와 동일 규약).
     예) 입력유형 역량을 전혀 요구하지 않는 사업에 'A 입력유형 적합 8점'은 평가 대상이 아니다.
     0점을 주면 '해당 없음'이 '나쁨'으로 바뀐다. None은 sum_with_denominator()가 분모에서 뺀다.

■ 조정 포인트
  ROADMAP_WEIGHT · LEVEL_MAX · SUB_CAPS 세 상수만 바꾸면 배점·가중치를 조절할 수 있다.

실행: python F3-2.py
"""

import importlib.util
import os
from functools import lru_cache
from pathlib import Path

import pandas as pd

BASE = Path(__file__).parent

# ════════════════════════════════════════════════════════════
# 0. 데이터 파일 경로 — 통합 폴더(app/data)와 개발 폴더 양쪽을 찾는다
# ════════════════════════════════════════════════════════════
DATA_DIRS = [
    Path(os.environ["SUNIC_DATA_DIR"]) if os.environ.get("SUNIC_DATA_DIR") else None,
    BASE,
    BASE.parent.parent / "data",   # core/scoring → core → app/data  (통합 폴더)
    BASE.parent / "app" / "data",
    Path.home() / "Desktop" / "app" / "data",
    Path.home() / "Desktop" / "A19" / "test_code" / "F2",
]


def _find(filename: str) -> Path:
    tried = []
    for d in DATA_DIRS:
        if d is None:
            continue
        p = d / filename
        tried.append(str(p))
        if p.exists():
            return p
    raise FileNotFoundError(f"{filename}을 찾을 수 없습니다:\n  - " + "\n  - ".join(tried))


# ════════════════════════════════════════════════════════════
# 1. 배점 (확정 배점표)
# ════════════════════════════════════════════════════════════
CAPS = {"조직역량적합도": 20, "역량전이가능성": 15, "부족역량수준": 15, "실행가능성": 5}

SUB_CAPS = {
    "조직역량적합도": {"A입력유형적합": 7, "A수행작업적합": 6, "B특허분류매칭": 4, "B로드맵연계": 3},
    "역량전이가능성": {"A범용역량전이": 6, "A입력유형전이": 6, "B유휴특허": 3},
    "부족역량수준": {"A미매칭역량": 5, "B특허미커버": 3, "C도메인인력부재": 7},
    "실행가능성": {"C도메인전문성": 5},
}

LEVEL_MAX = 5  # 조직보유수준 만점 (★1~5)

# 로드맵연계 등급 → 가중치. '미연계'는 0이지만 유휴특허 항목에서 따로 가점된다.
ROADMAP_WEIGHT = {"핵심": 1.0, "확장": 0.7, "개발 단계": 0.5, "보조": 0.3, "미연계": 0.0}

SYNTH_CAP_ID = "CAP_SYNTH"  # A 범용 역량 전이(증강·합성)의 기준 역량


# ════════════════════════════════════════════════════════════
# 2. 데이터 로드
# ════════════════════════════════════════════════════════════
@lru_cache(maxsize=1)
def load_std_caps() -> pd.DataFrame:
    """표준역량_정의.csv — 32개 표준역량. 조직 쪽 정보가 전부 여기 있다."""
    df = pd.read_csv(_find("표준역량_정의.csv"))
    df["조직보유수준"] = pd.to_numeric(df["조직보유수준"], errors="coerce").fillna(0)
    df["특허건수"] = pd.to_numeric(df["특허건수"], errors="coerce").fillna(0)
    return df.set_index("표준역량ID")


@lru_cache(maxsize=1)
def load_alias_map() -> dict:
    """역량어휘_매핑.csv — DB 자유 태그 177개 → 표준역량ID."""
    df = pd.read_csv(_find("역량어휘_매핑.csv"))
    return {str(r["DB태그"]).strip(): r["표준역량ID"] for _, r in df.iterrows()}


def cap_row(cap_id: str):
    std = load_std_caps()
    return std.loc[cap_id] if cap_id in std.index else None


def cap_name(cap_id: str) -> str:
    """표준역량ID → 사람이 읽는 이름. 모르는 ID는 ID를 그대로 돌려준다.

    matched/missing은 'CAP_3D' 같은 ID로 나온다. 대시보드나 F5 갭 리포트에
    그대로 뿌리면 사용자가 읽을 수 없다. 화면·리포트로 넘기기 직전에 이걸 통과시킨다.
    """
    r = cap_row(cap_id)
    return str(r["표준역량명"]) if r is not None else str(cap_id)


def cap_names(cap_ids) -> list:
    """ID 리스트를 이름 리스트로. F5 gap_data['missing']에 넣을 때 쓴다."""
    return [cap_name(c) for c in (cap_ids or [])]


# ── 업로드된 조직 프로필 (F1 standard_capability_levels) ────────────────
# None이면 표준역량_정의.csv 값을 쓴다(단독 실행·데모용).
#
# ⚠️ 왜 필요한가
#   CSV의 조직보유수준·특허건수·전담인력·로드맵연계는 특정 조직(샘플 데이터의 팀)을
#   미리 적어둔 스냅샷이다. 그것만 읽으면 다른 조직이 자기 파일을 올려도 조직계열
#   55점이 그대로 나온다. 업로드→파싱 결과가 점수에 반영되지 않으므로, 배포하면
#   바로 드러나는 문제다.
#   CSV에 남는 것은 조직과 무관한 정의뿐이다 — 표준역량명·역량성격·특허분류·전이출처.
_ORG_CAPS = None


def set_org_capabilities(caps: dict = None) -> None:
    """업로드 프로필의 표준역량 사실로 CSV 값을 덮어쓴다.

    Args:
        caps: F1 standard_capability_levels —
              {"CAP_VISION": {"level":5, "patent_count":6, "staff":[...],
                              "roadmap":"핵심"}, ...}
              보유수준만 있는 {"CAP_VISION": 5} 형태도 받는다.
              None이면 CSV로 되돌린다.

    ⚠️ caps가 주어지면 거기 없는 역량은 미보유(수준 0·특허 0·인력 없음)로 본다.
       CSV 값으로 메우면 업로드한 조직이 남의 스냅샷을 물려받는다.
    """
    global _ORG_CAPS
    if caps is None:
        _ORG_CAPS = None
    else:
        _ORG_CAPS = {}
        for k, v in caps.items():
            if not isinstance(v, dict):
                v = {"level": v}
            if v.get("level") is None:
                continue
            _ORG_CAPS[str(k)] = v
    _org_domain_level.cache_clear()   # 도메인 최고수준은 보유수준에서 나온다
    _idle_patent_caps.cache_clear()   # 유휴특허는 로드맵연계·특허건수에서 나온다


# 이전 이름 — 보유수준만 넘기던 호출부 호환용
set_org_levels = set_org_capabilities


def org_levels_source() -> str:
    """점수의 조직 정보가 어디서 왔는지. 화면에 표시해 혼동을 막는다."""
    return "표준역량_정의.csv" if _ORG_CAPS is None else "업로드 프로필"


def org_level(cap_id: str) -> float:
    """조직보유수준 0~5. 모르는 ID는 0."""
    if _ORG_CAPS is not None:
        v = _ORG_CAPS.get(cap_id)
        return float(v["level"]) if v else 0.0
    r = cap_row(cap_id)
    return float(r["조직보유수준"]) if r is not None else 0.0


def org_patent_count(cap_id: str) -> float:
    """이 역량을 뒷받침하는 조직 특허 건수."""
    if _ORG_CAPS is not None:
        v = _ORG_CAPS.get(cap_id)
        return float(v.get("patent_count") or 0) if v else 0.0
    r = cap_row(cap_id)
    return float(r["특허건수"]) if r is not None else 0.0


def org_roadmap(cap_id: str) -> str:
    """로드맵 연계 등급(핵심·확장·개발 단계·보조·미연계). 없으면 빈 문자열."""
    if _ORG_CAPS is not None:
        v = _ORG_CAPS.get(cap_id)
        return str(v.get("roadmap") or "") if v else ""
    r = cap_row(cap_id)
    if r is None or pd.isna(r["로드맵연계"]):
        return ""
    return str(r["로드맵연계"]).strip()


def org_has_staff(cap_id: str) -> bool:
    """이 역량에 배정된 전담인력이 있는지."""
    if _ORG_CAPS is not None:
        v = _ORG_CAPS.get(cap_id)
        return bool(v.get("staff")) if v else False
    r = cap_row(cap_id)
    return r is not None and pd.notna(r["전담인력"])


def is_held(cap_id: str) -> bool:
    return org_level(cap_id) >= 1


# ════════════════════════════════════════════════════════════
# 3. 전처리 — DB 필요역량태그 → 표준역량ID
# ════════════════════════════════════════════════════════════
def parse_required_capabilities(text) -> list:
    """'3D 재구성, 포인트클라우드, 영상정합' → ['CAP_MEASURE', 'CAP_DETECT', ...]

    구분자는 콤마만 쓴다('·'는 '검출·분할'처럼 복합 기술명의 일부다).
    이미 표준역량ID로 들어온 값(LLM 추천 후보)은 매핑 없이 그대로 통과시킨다.
    매핑 실패 태그는 조용히 버리지 않고 unmapped로 돌려준다.
    """
    ids, _ = parse_required_capabilities_detail(text)
    return ids


def parse_required_capabilities_detail(text) -> tuple:
    """(표준역량ID 리스트, 매핑 실패 태그 리스트)"""
    if text is None or (isinstance(text, float) and pd.isna(text)) or not str(text).strip():
        return [], []

    alias = load_alias_map()
    std_index = set(load_std_caps().index)

    ids, unmapped = [], []
    for raw in str(text).split(","):
        tag = raw.strip()
        if not tag:
            continue
        if tag in std_index:          # LLM 추천 후보는 이미 표준역량ID
            cid = tag
        elif tag in alias:
            cid = alias[tag]
        else:
            unmapped.append(tag)
            continue
        if cid not in ids:
            ids.append(cid)
    return ids, unmapped


# ════════════════════════════════════════════════════════════
# 4. 매칭 (F3-1 재사용)
# ════════════════════════════════════════════════════════════
def _load_f3_1():
    """F3-1을 불러온다.

    통합 폴더에서는 F3-1이 core/recommend/에, F3-2는 core/scoring/에 있다.
    파트 폴더가 다르므로 자기 폴더만 봐서는 못 찾는다. app 안에서 돌 때는
    core/paths.py에 경로 탐색을 맡기고, F3 폴더에서 단독 실행할 때는
    아래 폴백으로 옆 파일을 찾는다.
    """
    try:
        from core.paths import load_module
        return load_module("f3_1", "F3-1.py")
    except ImportError:
        pass

    for p in (BASE / "F3-1.py",                        # 단독 실행 (F3 파트 폴더)
              BASE.parent / "recommend" / "F3-1.py"):  # core/scoring → core/recommend
        if p.exists():
            spec = importlib.util.spec_from_file_location("f3_1", p)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError(
        f"F3-1.py가 필요합니다. 찾아본 곳: {BASE}, {BASE.parent / 'recommend'}")


match_capabilities = _load_f3_1().match_capabilities  # 로직 변경 없음


# ════════════════════════════════════════════════════════════
# 5. 세부항목 점수 — 모두 (요구역량 중 커버 비율) × 배점
#    평가 대상이 없으면 None (분모에서 제외)
# ════════════════════════════════════════════════════════════
def _ratio_by_level(cap_ids: list) -> float:
    """보유수준 가중 커버 비율. Σ(level/5) / n"""
    if not cap_ids:
        return 0.0
    return sum(min(org_level(c), LEVEL_MAX) / LEVEL_MAX for c in cap_ids) / len(cap_ids)


def _by_nature(cap_ids: list, nature: str) -> list:
    return [c for c in cap_ids
            if (r := cap_row(c)) is not None and str(r["역량성격"]).strip() == nature]


def score_input_type_fit(required: list) -> float:
    """A 입력 유형 적합 8점 — 요구된 '입력유형' 역량을 보유수준 가중으로 커버하는 비율."""
    targets = _by_nature(required, "입력유형")
    if not targets:
        return None
    return round(_ratio_by_level(targets) * SUB_CAPS["조직역량적합도"]["A입력유형적합"], 2)


def score_task_fit(required: list) -> float:
    """A 수행 작업 적합 7점 — 요구된 '수행작업' 역량 커버 비율."""
    targets = _by_nature(required, "수행작업")
    if not targets:
        return None
    return round(_ratio_by_level(targets) * SUB_CAPS["조직역량적합도"]["A수행작업적합"], 2)


def score_patent_class_match(required: list) -> float:
    """B 특허 분류 매칭 5점 — 요구역량 중 조직 특허가 붙어 있는 비율."""
    if not required:
        return None
    covered = sum(1 for c in required if org_patent_count(c) > 0)
    return round(covered / len(required) * SUB_CAPS["조직역량적합도"]["B특허분류매칭"], 2)


def score_roadmap_link(required: list) -> float:
    """B 로드맵 연계 4점 — 요구역량 중 '보유한 것'의 로드맵 등급 가중 평균.

    보유하지 않은 역량은 로드맵을 논할 수 없으므로 분모에서 뺀다.
    """
    held = [c for c in required if is_held(c)]
    if not held:
        return None
    w = [ROADMAP_WEIGHT.get(org_roadmap(c), 0.0) for c in held]
    return round(sum(w) / len(w) * SUB_CAPS["조직역량적합도"]["B로드맵연계"], 2)


def score_generic_transfer(missing: list) -> float:
    """A 범용 역량 전이(증강·합성) 6점.

    데이터 증강·합성은 '없는 도메인 데이터를 만들어 메우는' 범용 자산이다.
    메울 것이 없으면(미보유 요구역량 0개) 평가 대상이 아니므로 None.
    """
    if not missing:
        return None
    return round(min(org_level(SYNTH_CAP_ID), LEVEL_MAX) / LEVEL_MAX
                 * SUB_CAPS["역량전이가능성"]["A범용역량전이"], 2)


def score_input_type_transfer(missing: list) -> float:
    """A 입력유형 전이 6점 — 미보유 요구역량 중 '전이출처'가 조직 보유역량인 비율.

    전이출처 컬럼이 '이 역량은 어느 보유역량에서 전이 가능한가'를 담고 있다.
    """
    if not missing:
        return None
    ok = 0
    for c in missing:
        r = cap_row(c)
        if r is None:
            continue
        src = r["전이출처"]     # 전이출처는 역량 간 관계 정의 → CSV가 맞다
        if pd.notna(src) and is_held(str(src).strip()):
            ok += 1
    return round(ok / len(missing) * SUB_CAPS["역량전이가능성"]["A입력유형전이"], 2)


@lru_cache(maxsize=1)
def _idle_patent_caps() -> tuple:
    """로드맵 '미연계'인데 특허가 있는 보유역량 = 유휴 특허 (예: CAP_GEO 위치기반 7건).

    업로드 프로필이 주입되면 그 조직의 특허·로드맵으로 판정한다.
    """
    ids = list(_ORG_CAPS) if _ORG_CAPS is not None else list(load_std_caps().index)
    rows = [(cid, org_patent_count(cid)) for cid in ids
            if org_roadmap(cid) == "미연계" and org_patent_count(cid) > 0]
    return tuple(rows)


def score_idle_patent(required: list) -> float:
    """B 유휴 특허 6점 — 놀고 있는 특허가 이 사업에서 쓰이는 비율(특허건수 가중)."""
    idle = _idle_patent_caps()
    total = sum(n for _, n in idle)
    if total <= 0:
        return None
    used = sum(n for cid, n in idle if cid in required)
    return round(used / total * SUB_CAPS["역량전이가능성"]["B유휴특허"], 2)


def score_unmatched_gap(required: list, missing: list) -> float:
    """A 미매칭 역량 6점 — 부족이 적을수록 높다."""
    if not required:
        return None
    return round((1 - len(missing) / len(required))
                 * SUB_CAPS["부족역량수준"]["A미매칭역량"], 2)


def score_patent_uncovered(required: list) -> float:
    """B 특허 미커버 영역 4점 — 특허로 커버되는 비율(미커버가 적을수록 높다).

    ⚠️ '조직역량적합도 > B 특허분류 매칭 5점'과 같은 컬럼을 쓴다. 배점표가 같은
       데이터를 가점(적합도)과 감점(부족역량) 양쪽에 배치했기 때문이며, 두 항목은
       상관이 1.0이 된다. 확정 배점표를 그대로 따른 결과이므로 유지하되,
       변별력 관점에서는 F2-5의 '사업성 6점 = 시장성 복사'와 같은 성격이다.
    """
    if not required:
        return None
    covered = sum(1 for c in required if org_patent_count(c) > 0)
    return round(covered / len(required) * SUB_CAPS["부족역량수준"]["B특허미커버"], 2)


def score_domain_staff(required: list) -> float:
    """C 도메인 인력 부재 8점 — 요구역량 중 전담인력이 배정된 비율."""
    if not required:
        return None
    staffed = sum(1 for c in required if org_has_staff(c))
    return round(staffed / len(required) * SUB_CAPS["부족역량수준"]["C도메인인력부재"], 2)


@lru_cache(maxsize=1)
def _org_domain_level() -> float:
    """조직이 보유한 도메인 전문성의 '최고 수준'.

    평균이 아니라 최댓값을 쓴다. 표준역량에는 조직과 무관한 도메인(농업·환경·식품)이
    같이 들어 있어서 평균을 내면 수의 도메인 ★4가 1.0으로 희석된다
    (2026-08-06 실측: 도메인 4개 중 CAP_DOM_VET만 ★4, 나머지 0 → 평균 1.0).
    '도메인 전문가를 보유했는가'를 묻는 항목이므로 최고 수준이 맞다.
    """
    std = load_std_caps()
    # 역량성격(도메인/입력유형/…)은 분류 체계라 CSV가 맞지만, 보유수준은 조직마다
    # 다르므로 org_level()을 통과시켜 업로드 프로필이 있으면 그 값을 쓴다.
    lv = [org_level(cid) for cid, r in std.iterrows()
          if str(r["역량성격"]).strip() == "도메인"]
    return max(lv) if lv else 0.0


def score_domain_expertise(required: list) -> float:
    """C 도메인 전문성 5점 (실행 가능성).

    사업이 도메인 역량을 요구하면 그 커버 비율로, 요구하지 않으면 조직 전체의
    도메인 전문성 수준으로 매긴다(장다연 노트의 '조직 상수' 성격 유지).
    """
    cap = SUB_CAPS["실행가능성"]["C도메인전문성"]
    targets = _by_nature(required, "도메인")
    if targets:
        return round(_ratio_by_level(targets) * cap, 2)
    return round(min(_org_domain_level(), LEVEL_MAX) / LEVEL_MAX * cap, 2)


# ════════════════════════════════════════════════════════════
# 6. 통합 — 4개 항목 (F3-4에서 그대로 사용)
# ════════════════════════════════════════════════════════════
def _agg(subs: dict):
    """세부항목 dict → 항목 점수. 전부 None이면 None(=평가 대상 없음)."""
    vals = [v for v in subs.values() if v is not None]
    return round(sum(vals), 2) if vals else None


def calculate_org_series_scores(required_caps, org_context: dict = None,
                                return_detail: bool = False):
    """조직계열 4개 항목 점수.

    Args:
        required_caps : DB '필요역량태그' 원문(str) 또는 표준역량ID 리스트
        org_context   : 미사용. 조직 보유수준은 set_org_levels()로 주입한다
                        (업로드 프로필이 없으면 표준역량_정의.csv를 쓴다).
                        F3-4 호환을 위해 인자만 유지한다.
        return_detail : True면 (scores, detail) 반환

    Returns:
        {"조직역량적합도": float|None, "역량전이가능성": float|None,
         "부족역량수준": float|None, "실행가능성": float|None}
    """
    if isinstance(required_caps, str) or required_caps is None:
        required, unmapped = parse_required_capabilities_detail(required_caps)
    else:
        required, unmapped = list(required_caps), []

    pool = list(_ORG_CAPS) if _ORG_CAPS is not None else list(load_std_caps().index)
    org_caps = [cid for cid in pool if is_held(cid)]
    m = match_capabilities(org_caps, required)
    matched, missing = m["matched"], m["missing"]

    detail = {
        "조직역량적합도": {
            "A입력유형적합": score_input_type_fit(required),
            "A수행작업적합": score_task_fit(required),
            "B특허분류매칭": score_patent_class_match(required),
            "B로드맵연계": score_roadmap_link(required),
        },
        "역량전이가능성": {
            "A범용역량전이": score_generic_transfer(missing),
            "A입력유형전이": score_input_type_transfer(missing),
            "B유휴특허": score_idle_patent(required),
        },
        "부족역량수준": {
            "A미매칭역량": score_unmatched_gap(required, missing),
            "B특허미커버": score_patent_uncovered(required),
            "C도메인인력부재": score_domain_staff(required),
        },
        "실행가능성": {
            "C도메인전문성": score_domain_expertise(required),
        },
    }
    scores = {k: _agg(v) for k, v in detail.items()}

    if return_detail:
        return scores, {"sub_scores": detail, "matched": matched, "missing": missing,
                        "required_ids": required, "unmapped_tags": unmapped}
    return scores


def sum_with_denominator(scores: dict, caps: dict = None) -> tuple:
    """None인 항목을 분모에서 빼고 (획득점수, 배점합)을 돌려준다. F2-5와 동일 규약."""
    caps = caps or CAPS
    got = sum(v for v in scores.values() if v is not None)
    den = sum(caps[k] for k, v in scores.items() if v is not None and k in caps)
    return round(got, 1), den


# ════════════════════════════════════════════════════════════
# 7. 검증
# ════════════════════════════════════════════════════════════
def _load_db() -> pd.DataFrame:
    return pd.read_excel(_find("신사업_DB.xlsx"))


def run_all() -> list:
    df = _load_db()
    out = []
    for _, row in df.iterrows():
        scores, det = calculate_org_series_scores(row.get("필요역량태그", ""),
                                                  return_detail=True)
        got, den = sum_with_denominator(scores)
        out.append({
            "아이디어ID": row.get("아이디어ID", "unknown"),
            "아이디어명": str(row.get("아이디어명", ""))[:22],
            **{k: (v if v is not None else None) for k, v in scores.items()},
            "소계": got, "분모": den,
            "미매핑태그": len(det["unmapped_tags"]),
        })
    return out


def check_1_배점초과(rows: list):
    print("\n[체크 1] 배점 초과 — 자동")
    ok = True
    for label, cap in CAPS.items():
        over = [r for r in rows if r[label] is not None and r[label] > cap + 1e-6]
        if over:
            ok = False
            print(f"  ❌ {label}: {len(over)}건이 {cap}점 초과 — {[r['아이디어ID'] for r in over][:5]}")
    print("  ✅ 전항목 배점 이내" if ok else "  → 위 항목 로직 재확인 필요")


def check_2_세부합계(rows: list):
    print("\n[체크 2] 세부항목 합 = 항목 배점 — 자동")
    ok = True
    for item, subs in SUB_CAPS.items():
        if sum(subs.values()) != CAPS[item]:
            ok = False
            print(f"  ❌ {item}: 세부 합 {sum(subs.values())} ≠ 배점 {CAPS[item]}")
    print(f"  전체 합계: {sum(CAPS.values())}점 (시장계열 45점과 합쳐 100점)")
    print("  ✅ 세부항목 합 일치" if ok else "  → SUB_CAPS 재확인 필요")


def check_3_미매핑(rows: list):
    print("\n[체크 3] DB 태그 매핑 실패 — 자동")
    bad = [r for r in rows if r["미매핑태그"] > 0]
    if bad:
        print(f"  ⚠️ {len(bad)}건에서 매핑 실패 태그 발견 — 역량어휘_매핑.csv 보강 필요")
        for r in bad[:5]:
            print(f"    - {r['아이디어ID']} {r['아이디어명']}: {r['미매핑태그']}개")
    else:
        print("  ✅ 50건 전부 매핑 성공")


def check_4_변별력(rows: list):
    print("\n[체크 4] 변별력 — 직접 눈으로 확인")
    s = pd.Series([r["소계"] for r in rows])
    print(f"  소계 평균 {s.mean():.1f} · 표준편차 {s.std():.1f} · 최소 {s.min()} · 최대 {s.max()}")
    dup = s.value_counts()
    print(f"  최빈값 {dup.index[0]}점이 {dup.iloc[0]}건 "
          f"({dup.iloc[0] / len(s) * 100:.0f}%)")
    for label in CAPS:
        vals = [r[label] for r in rows if r[label] is not None]
        none_n = len(rows) - len(vals)
        sd = pd.Series(vals).std() if vals else 0
        print(f"  {label:9} 표준편차 {sd:.2f} · 미평가(None) {none_n}건")


if __name__ == "__main__":
    std = load_std_caps()
    print(f"표준역량 {len(std)}개 · 조직 보유 {sum(1 for c in std.index if is_held(c))}개 "
          f"· 유휴특허 {_idle_patent_caps()}")
    print(f"배점: " + " / ".join(f"{k} {v}" for k, v in CAPS.items())
          + f"  = {sum(CAPS.values())}점")

    rows = run_all()
    df_out = pd.DataFrame(rows)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 160)
    print("\n" + df_out.to_string(index=False))

    check_1_배점초과(rows)
    check_2_세부합계(rows)
    check_3_미매핑(rows)
    check_4_변별력(rows)

    df_out.to_csv("f3_2_scores.csv", index=False, encoding="utf-8-sig")
    print("\n저장 완료: f3_2_scores.csv")
