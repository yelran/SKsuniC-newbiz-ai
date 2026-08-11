"""
F4-4. 아이디어 입력 전용 100점 채점 — 7A단계 (담당: 김민주)

7B(DB 매칭 후보)는 F3-2(조직계열) + F2-5(시장계열)로 채점하는데, F2-5는 DB 행의
실측 시장규모·진입장벽 텍스트를 파싱해서 쓴다. 7A(사용자가 직접 입력한 아이디어,
6C의 F4-3 신규 제안 포함)는 DB에 없는 아이디어라 그 실측 데이터가 없다. 그래서 7A는:

  - 조직계열: F3-2.calculate_org_series_scores()를 그대로 재사용한다
    (점수 공식·배점 변경 없음 — F3-2가 바뀌면 7A도 자동으로 같이 바뀐다).
    예외 하나: 역량전이가능성이 미평가(None) 또는 0점(=요구역량을 조직이
    32개 표준역량 축 안에서 100% 보유해 전이할 대상이 없음)이면서, F4-1이
    32개에도 없어서 자유서술로 낸 unmatched_capabilities가 있으면, 항목당
    고정 2점(최대 3개=6점)을 역량전이가능성에 가점한다(2026-08-09 추가).
    LLM은 "목록 밖 역량이 몇 개 있는지"만 판단하고 "몇 점인지"는 고정 상수가
    정한다 — required_caps엔 안 섞어서 조직역량적합도·부족역량수준·실행가능성은
    영향받지 않는다. 적용 여부는 결과의 transfer_bonus 키로 노출한다.
  - 시장계열: LLM이 "시장규모(억달러)·진입장벽(5단계)"만 추정하고, 그 추정값을
    F2-5의 실제 점수 계산 함수(calculate_market_series_scores)에 그대로 넣어
    점수를 낸다. LLM은 숫자를 "추정"만 하고 "채점"은 F2-5가 한다 — 그래야 7A와
    7B가 같은 배점 공식을 공유하고, LLM이 점수를 직접 지어내는 것보다 일관적이다.
    항목 개수·이름·배점은 이 파일에 하드코딩하지 않는다(f3_2.CAPS/f2_5.CAPS를
    그대로 읽음) — F2-5가 항목을 추가·삭제해도(예: 경쟁강도 부활, 2026-08-06)
    이 파일은 손 안 대도 따라간다.

AI 추정치라는 걸 화면에서 명확히 표시해야 한다(DB 실측과 혼동 방지) —
market_estimate.is_estimated=True와 rationale을 그대로 노출할 것.

에러 처리: OPENAI_API_KEY가 없거나 호출이 실패하면 시장계열 항목을 전부 None으로
둔다(0점이 아니라 "미평가" — F2-5/F3-2와 동일 규약). 조직계열 점수만으로도 화면은 뜬다.

calculate_idea_score(required_caps, idea_context, org_context=None, return_detail=False) -> dict
"""

import importlib.util
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

BASE = Path(__file__).parent

# F3-2·F2-5는 core/scoring/에 있다. F4-4는 core/idea_fit/에 있어서 폴더가 다르다.
SEARCH_DIRS = [
    Path(os.environ["SUNIC_CODE_DIR"]) if os.environ.get("SUNIC_CODE_DIR") else None,
    BASE,
    BASE.parent / "scoring",            # core/idea_fit → core/scoring
    Path.home() / "Desktop" / "app" / "core" / "scoring",
]


def _load(name: str, filename: str):
    """F3-4.py와 같은 로더 패턴 — core.paths가 있으면 그걸로 캐시 공유, 없으면 직접 탐색."""
    try:
        from core.paths import load_module
        return load_module(name, filename)
    except ImportError:
        pass

    tried = []
    for d in SEARCH_DIRS:
        if d is None:
            continue
        p = d / filename
        tried.append(str(p))
        if p.exists():
            spec = importlib.util.spec_from_file_location(name, p)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError(f"{filename}을 찾을 수 없습니다:\n  - " + "\n  - ".join(tried))


f2_5 = _load("f2_5", "F2-5.py")
f3_2 = _load("f3_2", "F3-2.py")


# ════════════════════════════════════════════════════════════
# 1. 배점 — F3-2/F2-5 값을 그대로 합쳐서 만든다(하드코딩 금지, F3-4와 동일 원칙)
# ════════════════════════════════════════════════════════════
CAPS = {**f3_2.CAPS, **f2_5.CAPS}
TOTAL_CAP = sum(CAPS.values())

OPENAI_MODEL = os.environ.get("F4_4_OPENAI_MODEL", "gpt-5.6-terra")


# ════════════════════════════════════════════════════════════
# 2. LLM 시장 추정 (숫자만 추정, 채점은 F2-5가 한다)
# ════════════════════════════════════════════════════════════
class MarketEstimate(BaseModel):
    market_size_usd: float = Field(
        ..., description="추정 목표 시장 규모, 억달러 단위(예: 120.5). 과장하지 말고 "
                          "유사 산업의 실제 시장 규모를 참고해 보수적으로 추정."
    )
    # F2-5.ENTRY_BARRIER_TABLE이 5단계라 여기도 맞춘다. 3단계로 두면 LLM이
    # "중상"/"중하"를 답할 방법이 없어서 정밀도가 떨어진다(에러는 안 나지만).
    entry_barrier: Literal["상", "중상", "중", "중하", "하"] = Field(..., description="추정 진입장벽 수준(5단계)")
    rationale: str = Field(..., description="이렇게 추정한 근거 2~3문장 — 참고한 유사 시장/사업 사례 등")


def _get_openai_client():
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        # Streamlit Cloud 등 배포 환경에서는 secrets.toml에만 키가 있고
        # os.environ에는 없을 수 있으므로 st.secrets도 확인한다.
        try:
            import streamlit as st

            api_key = st.secrets.get("OPENAI_API_KEY")
        except Exception:
            api_key = None
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY가 설정되어 있지 않습니다.")
    return OpenAI(api_key=api_key)


def _estimate_market_inputs(idea_context: dict) -> MarketEstimate:
    client = _get_openai_client()
    system_prompt = (
        "너는 시장조사 애널리스트다. 사용자가 설명한 신사업 아이디어를 보고, 실제 시장 "
        "데이터가 없는 상태에서 목표 시장 규모와 진입장벽을 합리적으로 추정한다. "
        "과장하지 말고, 유사한 산업의 실제 사례를 참고해 보수적으로 추정해라."
    )
    user_prompt = (
        f"진출 산업: {idea_context.get('industry', '')}\n"
        f"해결 문제: {idea_context.get('problem', '')}\n"
        f"목표 시장: {idea_context.get('market', '')}\n"
        f"목표 고객: {idea_context.get('customer', '')}\n\n"
        "이 아이디어의 목표 시장 규모(억달러)와 진입장벽 수준(상/중상/중/중하/하 5단계 중 하나)을 추정해줘."
    )
    resp = client.responses.parse(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        text_format=MarketEstimate,
    )
    return resp.output_parsed


def _score_market_series(idea_context: dict, org_context: dict) -> tuple:
    """(시장계열 항목 점수 dict, market_estimate 정보 dict)

    실패해도 예외를 던지지 않는다 — 조직계열 점수만으로도 화면이 떠야 한다.
    항목 이름·개수를 하드코딩하지 않고 f2_5.CAPS에서 그대로 가져온다 —
    F2-5가 항목을 추가·삭제해도(예: 경쟁강도 부활) 이 파일을 안 고쳐도 따라간다.
    """
    none_scores = {k: None for k in f2_5.CAPS}
    try:
        estimate = _estimate_market_inputs(idea_context)
    except Exception as e:
        return none_scores, {"is_estimated": True, "error": str(e)}

    market_data = {"market_size_usd": estimate.market_size_usd, "entry_barrier": estimate.entry_barrier}
    scores = f2_5.calculate_market_series_scores(market_data, org_context)
    return scores, {
        "is_estimated": True,
        "market_size_usd": estimate.market_size_usd,
        "entry_barrier": estimate.entry_barrier,
        "rationale": estimate.rationale,
        "error": None,
    }


# ════════════════════════════════════════════════════════════
# 3. 목록 밖 역량 보너스 — 역량전이가능성
#
# 요구역량을 조직이 32개 표준역량 축 안에서 100% 보유(missing=[])하면 전이할
# 부족역량 자체가 없어서 역량전이가능성이 미평가(None)가 된다 — F3-2의
# score_generic_transfer()/score_input_type_transfer()가 `if not missing: return None`.
# (0점만 확인하면 이 경우가 통째로 빠져 보너스가 안 붙었다 — 2026-08-10 수정)
# 근데 F4-1이 "32개에도 없어서" 자유서술로 낸 unmatched_capabilities는 정의상
# 조직이 못 가진 게 확실한 역량이라, 이것도 부족역량의 일종으로 볼 수 있다.
#
# 다만 required_caps에 섞어 넣진 않는다 — 섞으면 조직역량적합도·부족역량수준·
# 실행가능성까지 같은 리스트를 나눠 쓰므로 같이 오염된다(2026-08-09 논의).
# 대신 역량전이가능성에만 고정 배점으로 소액 가점한다. LLM은 "몇 개 있는지"만
# 판단하고 "몇 점인지"는 절대 안 정한다(항목당 고정 UNMATCHED_BONUS_PER_POINT) —
# F4-4 전체의 "LLM은 추정, 점수는 공식" 원칙을 여기서도 유지한다.
# ════════════════════════════════════════════════════════════
UNMATCHED_BONUS_PER_ITEM = 2      # 항목당 고정 점수 (LLM이 정하지 않음)
UNMATCHED_BONUS_MAX_ITEMS = 3     # 최대 반영 개수 (F4-1의 MAX_UNMATCHED_CAPABILITIES와 별개 캡)
TRANSFER_CATEGORY = "역량전이가능성"
TRANSFER_BONUS_SUBITEM = "A입력유형전이"


def _apply_unmatched_transfer_bonus(
    org_scores: dict, org_detail: dict, unmatched_capabilities: list | None
) -> dict:
    """표준 목록 밖 역량을 제한적으로 반영하고 적용 근거를 구조화해 반환한다.

    F3-2는 미보유 표준역량이 없으면 역량전이가능성을 0이 아니라 None으로 둔다.
    전달본은 정확히 0인 경우만 확인해 설명과 달리 보너스가 빠질 수 있었으므로
    None과 0을 모두 대상에 포함한다. 상세점수 합과 항목점수가 어긋나지 않도록
    보너스는 A입력유형전이 슬롯에 별도 출처와 함께 기록한다.
    """
    items = [str(item).strip() for item in (unmatched_capabilities or []) if str(item).strip()]
    base_score = org_scores.get(TRANSFER_CATEGORY)
    can_apply = base_score is None or float(base_score) == 0.0
    if not items or not can_apply:
        return {"applied": False, "amount": 0.0, "item_count": 0, "items": []}

    item_count = min(len(items), UNMATCHED_BONUS_MAX_ITEMS)
    amount = float(min(item_count * UNMATCHED_BONUS_PER_ITEM, CAPS[TRANSFER_CATEGORY]))
    org_scores[TRANSFER_CATEGORY] = amount
    sub_scores = org_detail.setdefault("sub_scores", {}).setdefault(TRANSFER_CATEGORY, {})
    sub_scores[TRANSFER_BONUS_SUBITEM] = amount
    return {
        "applied": True,
        "amount": amount,
        "item_count": item_count,
        "items": items[:item_count],
        "subitem": TRANSFER_BONUS_SUBITEM,
        "is_provisional": True,
    }


# ════════════════════════════════════════════════════════════
# 4. 메인 함수
# ════════════════════════════════════════════════════════════
def calculate_idea_score(required_caps, idea_context: dict, org_context: dict = None,
                         return_detail: bool = False, unmatched_capabilities: list = None) -> dict:
    """
    Args:
        required_caps: 표준역량ID 리스트 (F4-1 요구역량을 표준역량 축으로 직접 낸 것)
        idea_context:  {"industry","problem","market","customer"} — LLM 시장 추정용 원문
        org_context:   F2-5가 쓰는 조직 플래그(has_commercialization_experience 등).
                        조직계열 보유수준은 F3-2.set_org_capabilities()로 이미 주입돼 있어야 한다.
        return_detail: True면 조직계열 세부 항목(F3-2 detail)까지 반환
        unmatched_capabilities: F4-1이 32개 표준역량 축에서 못 찾은 요구역량(자유서술).
                        역량전이가능성이 0점이거나 미평가(None)일 때만, 항목당 고정
                        점수로 소액 가점한다.

    Returns:
        {"scores": {8개 항목: float|None}, "total_score": float, "raw_score": float,
         "denominator": int, "market_estimate": {...} | None,
         "transfer_bonus": {...}, "detail": {...}(옵션)}
    """
    org_context = org_context or {}

    org_scores, org_detail = f3_2.calculate_org_series_scores(
        required_caps, org_context, return_detail=True)
    transfer_bonus = _apply_unmatched_transfer_bonus(
        org_scores, org_detail, unmatched_capabilities
    )
    market_scores, market_estimate = _score_market_series(idea_context, org_context)

    scores = {**org_scores, **market_scores}
    raw, den = f2_5.sum_with_denominator(scores, CAPS)
    total = round(raw / den * 100, 1) if den else 0.0

    result = {
        "scores": scores, "total_score": total, "raw_score": raw, "denominator": den,
        "market_estimate": market_estimate,
        "transfer_bonus": transfer_bonus,
    }
    if return_detail:
        result["detail"] = org_detail
    return result


if __name__ == "__main__":
    import json

    print("배점: " + " / ".join(f"{k} {v}" for k, v in CAPS.items()))
    print(f"합계 {TOTAL_CAP}점 (F3-2 {sum(f3_2.CAPS.values())} + F2-5 {sum(f2_5.CAPS.values())})\n")

    # API 키 없이 실행하면 시장계열 None(=미평가), 조직계열만 채점되는 경로를 확인한다.
    demo_required = ["CAP_VISION", "CAP_MEASURE"]
    demo_idea = {
        "industry": "수의·반려동물 헬스케어",
        "problem": "동물병원 X-ray 판독을 원격으로 실시간 지원",
        "market": "국내 동물병원 시장",
        "customer": "1~2인 소규모 동물병원",
    }
    result = calculate_idea_score(demo_required, demo_idea, {}, return_detail=True)
    print(json.dumps(result, ensure_ascii=False, indent=2))
