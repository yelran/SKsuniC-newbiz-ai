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

SEARCH_DIRS = [
    Path(os.environ["SUNIC_CODE_DIR"]) if os.environ.get("SUNIC_CODE_DIR") else None,
    BASE,
    BASE.parent / "scoring",            # core/idea_fit → core/scoring
    Path.home() / "Desktop" / "app" / "core" / "scoring",
]


def _load(name: str, filename: str):
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
# 1. 배점 
# ════════════════════════════════════════════════════════════
CAPS = {**f3_2.CAPS, **f2_5.CAPS}
TOTAL_CAP = sum(CAPS.values())

OPENAI_MODEL = os.environ.get("F4_4_OPENAI_MODEL", "gpt-5.6-terra")


# ════════════════════════════════════════════════════════════
# 2. LLM 시장 추정 
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
# ════════════════════════════════════════════════════════════
UNMATCHED_BONUS_PER_ITEM = 2      # 항목당 고정 점수 (LLM이 정하지 않음)
UNMATCHED_BONUS_MAX_ITEMS = 3     # 최대 반영 개수
TRANSFER_CATEGORY = "역량전이가능성"
TRANSFER_BONUS_SUBITEM = "A입력유형전이"


def _apply_unmatched_transfer_bonus(
    org_scores: dict, org_detail: dict, unmatched_capabilities: list | None
) -> dict:
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
