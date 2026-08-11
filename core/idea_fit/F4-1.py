import importlib.util
import os
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field

try:
    # 로컬 개발용 .env에서 OPENAI_API_KEY 등을 읽어온다.
    # 배포(Streamlit Cloud)에서는 st.secrets를 쓰므로 python-dotenv가 없어도
    # 동작해야 한다. 여기서 그냥 import하면 패키지가 없는 팀원 PC에서
    # F4 화면 전체가 로드조차 되지 않는다.
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

BASE = Path(__file__).parent


# ---------------------------------------------------------------
# 0. 역량 정규화 
# ---------------------------------------------------------------

from f4_common import (  # noqa: E402
    CAP_ID_VALUES,
    CAP_NAME_BY_ID,
    CAPABILITY_CHOICES,
    CAPABILITY_MENU,
    CapabilityIdLiteral,
    load_f1,
)

f1 = load_f1()

CAPABILITY_DEFINITIONS = [
    {"capability_id": c["id"], "name": c["name"], "desc": c.get("desc", "")}
    for c in CAPABILITY_CHOICES
]

EMBEDDING_MODEL_NAME = "jhgan/ko-sroberta-multitask"  # main.py(F2-2)와 동일 모델 — 로딩 캐시 공유 목적


EXTRA_KEYWORDS: dict[str, list[str]] = {}


def _capability_reference_text(cap: dict) -> str:
    kws = EXTRA_KEYWORDS.get(cap["capability_id"], [])
    desc = cap.get("desc", "")
    return " ".join(part for part in (cap["name"], desc, " ".join(kws)) if part)


@lru_cache(maxsize=1)
def _get_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@lru_cache(maxsize=1)
def _capability_embeddings():
    model = _get_embedding_model()
    texts = [_capability_reference_text(c) for c in CAPABILITY_DEFINITIONS]
    return model.encode(texts, convert_to_tensor=True)


def keyword_hits(text: str, cap_id: str) -> int:
    kws = EXTRA_KEYWORDS.get(cap_id, [])
    return sum(1 for kw in kws if kw in text)


def map_text_to_capabilities(text: str, top_k: int = 3, sim_threshold: float = 0.35,
                              keyword_bonus: float = 0.15) -> list:
    """자유서술 텍스트 -> F1 capability_id 리스트 정규화.

    Returns: [{"capability_id","name","score","similarity","keyword_hits"}, ...]
             score(유사도+키워드 보너스) 내림차순, threshold 미달·키워드 无 항목은 제외.
    """
    if not text or not text.strip():
        return []

    from sentence_transformers import util
    model = _get_embedding_model()
    cap_emb = _capability_embeddings()
    query_emb = model.encode(text, convert_to_tensor=True)
    sims = util.cos_sim(query_emb, cap_emb)[0].tolist()

    results = []
    for cap, sim in zip(CAPABILITY_DEFINITIONS, sims):
        hits = keyword_hits(text, cap["capability_id"])
        score = sim + keyword_bonus * hits
        results.append({
            "capability_id": cap["capability_id"],
            "name": cap["name"],
            "score": round(score, 4),
            "similarity": round(sim, 4),
            "keyword_hits": hits,
        })

    results.sort(key=lambda r: r["score"], reverse=True)
    passed = [r for r in results if r["score"] >= sim_threshold or r["keyword_hits"] > 0]
    return passed[:top_k] if top_k else passed


def confidence_level(results: list) -> str:
    
    if not results:
        return "none"
    top = results[0]
    if top["keyword_hits"] > 0 or top["score"] >= 0.55:
        return "high"
    if top["score"] >= 0.35:
        return "medium"
    return "low"


# ---------------------------------------------------------------
# 1. 설정
# ---------------------------------------------------------------
OPENAI_MODEL = os.environ.get("F4_1_OPENAI_MODEL", "gpt-5.6-terra")

LOW_CONFIDENCE_LEVELS = {"medium", "low", "none"}


_CAP_ID_VALUES = CAP_ID_VALUES
_CAP_NAME_BY_ID = CAP_NAME_BY_ID


MAX_UNMATCHED_CAPABILITIES = 2


# ---------------------------------------------------------------
# 2. 입출력 스키마 (pydantic)
# ---------------------------------------------------------------
class IdeaInput(BaseModel):
    industry: str = Field(..., description="진출 산업")
    problem: str = Field(..., description="해결하려는 문제")
    market: str = Field(..., description="목표 시장")
    customer: str = Field(..., description="목표 고객")


class LLMCapabilityEstimate(BaseModel):
    """OpenAI structured output 스키마 — capability_id는 표준역량 32개 값으로만 제한."""
    capability_ids: list[CapabilityIdLiteral] = Field(
        default_factory=list,
        description="이 아이디어를 실행하는 데 필요할 것으로 추정되는 표준역량 ID들",
    )
    unmatched_capabilities: list[str] = Field(
        default_factory=list,
        description=(
            "표준역량 32개 목록 중 정말로 해당하는 게 하나도 없을 때만 적는, 이 "
            "아이디어에 필요한 역량의 짧은 이름(명사구). 애매하면 비워 둔다. "
            f"최대 {MAX_UNMATCHED_CAPABILITIES}개."
        ),
    )
    rationale: str = Field(..., description="왜 이 역량들이 필요하다고 추정했는지, 1~3문장 설명")


# ---------------------------------------------------------------
# 3. OpenAI 클라이언트 (fallback 전용)
# ---------------------------------------------------------------
@lru_cache(maxsize=1)
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


def _llm_estimate_capabilities(idea: IdeaInput) -> LLMCapabilityEstimate:
    client = _get_openai_client()
    capability_menu = "\n".join(f"- {cid}: {name}" for cid, name in _CAP_NAME_BY_ID.items())

   
    system_prompt = (
        "너는 신사업 진입 전략 분석가다. 사용자가 설명한 신규 아이디어를 실행하려면 "
        "아래 표준역량 카테고리 중 어떤 것들이 필요할지 추정한다. 이 조직이 지금 "
        "무엇을 갖고 있는지는 모른다고 가정해라 — 갖고 있는지 여부와 상관없이, 이 "
        "아이디어 자체가 실행에 필요로 하는 역량만 순수하게 판단해라.\n\n"
        "규칙:\n"
        "1. 반드시 아래 목록에 있는 capability_id만 capability_ids에 사용해야 하며, "
        "목록에 없는 새 카테고리를 만들어내면 안 된다. 확신이 없으면 관련성이 높은 "
        "것만 최대 3개까지만 고른다.\n"
        "2. 목록 32개 중 정말로 해당하는 게 하나도 없는 필요역량이 있을 때만 "
        f"unmatched_capabilities에 짧은 이름으로 적어라(최대 {MAX_UNMATCHED_CAPABILITIES}개). "
        "억지로 채우지 마라 — 목록 안에서 조금이라도 겹치는 게 있으면 그쪽을 우선한다.\n\n"
        f"[표준역량 카테고리 목록]\n{capability_menu}"
    )
    user_prompt = (
        f"진출 산업: {idea.industry}\n"
        f"해결하려는 문제: {idea.problem}\n"
        f"목표 시장: {idea.market}\n"
        f"목표 고객: {idea.customer}\n\n"
        "이 아이디어를 실행하는 데 필요한 역량 카테고리를 추정해줘."
    )

    resp = client.responses.parse(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        text_format=LLMCapabilityEstimate,
    )
    estimate = resp.output_parsed
    estimate.unmatched_capabilities = estimate.unmatched_capabilities[:MAX_UNMATCHED_CAPABILITIES]
    return estimate


# ---------------------------------------------------------------
# 4. 메인 함수
# ---------------------------------------------------------------
def extract_idea_requirements(user_input: dict) -> dict:
    """
    Args:
        user_input: {"industry", "problem", "market", "customer"}

    Returns:
        {
          "input": {...원본 4필드...},
          "target_industry": str,
          "required_capabilities": [
              {"capability_id", "name", "confidence": float|None, "source": "keyword"|"embedding"|"llm"}
          ],
          # 2026-08-09부터 표준역량 축(CAP_*, F3-2가 실제로 채점하는 축)을 직접 낸다.
          # main.py는 이제 이 값을 번역 없이 그대로 F4-4에 넘기면 된다.
          "required_capability_ids": [str, ...],
          "required_capability_names": [str, ...],   # 위 ID의 표준역량명(화면 표시용)
          "unmatched_capabilities": [str, ...],      # 표준역량 32개에 없는 것 — 채점엔 안 씀, 표시 전용
          "confidence": "high"|"medium"|"low"|"none",   # 1차(로컬) 매핑 신뢰도
          "used_llm_fallback": bool,
          "llm_rationale": str|None,
        }
    """
    idea = IdeaInput(**user_input)

    # 1차: 로컬 임베딩+키워드 매핑 (문제·고객 설명이 요구역량 정보를 가장 많이 담고 있음)
    combined_text = f"{idea.problem} {idea.customer} {idea.market}"
    local_results = map_text_to_capabilities(combined_text)
    confidence = confidence_level(local_results)

    required = [
        {
            "capability_id": r["capability_id"],
            "name": r["name"],
            "confidence": r["score"],
            "source": "keyword" if r["keyword_hits"] else "embedding",
        }
        for r in local_results
    ]

    used_llm_fallback = False
    llm_rationale = None
    unmatched_capabilities: list[str] = []

    # 2차: 로컬 매핑 신뢰도가 낮을 때만("신규 아이디어 추정치") OpenAI 보완
    if confidence in LOW_CONFIDENCE_LEVELS:
        try:
            estimate = _llm_estimate_capabilities(idea)
            used_llm_fallback = True
            llm_rationale = estimate.rationale
            unmatched_capabilities = estimate.unmatched_capabilities
            existing_ids = {r["capability_id"] for r in required}
            for cid in estimate.capability_ids:
                if cid not in _CAP_NAME_BY_ID:
                    continue  # 이중 방어: 화이트리스트 밖 값은 무시
                if cid in existing_ids:
                    continue
                required.append({
                    "capability_id": cid,
                    "name": _CAP_NAME_BY_ID[cid],
                    "confidence": None,  # LLM 추정치는 로컬 유사도 점수가 없음
                    "source": "llm",
                })
                existing_ids.add(cid)
        except Exception as e:
            print(f" OpenAI fallback 실패, 로컬 매핑 결과만 사용합니다: {e}")

    return {
        "input": idea.model_dump(),
        "target_industry": idea.industry,
        "required_capabilities": required,
        "required_capability_ids": [r["capability_id"] for r in required],
        "required_capability_names": [r["name"] for r in required],
        "unmatched_capabilities": unmatched_capabilities,
        "confidence": confidence,
        "used_llm_fallback": used_llm_fallback,
        "llm_rationale": llm_rationale,
    }


if __name__ == "__main__":
    import json

    samples = [
        {
            "industry": "수의·반려동물 헬스케어",
            "problem": "동물병원의 X-ray 판독을 원격으로도 실시간 지원해, 야간·소규모 병원의 진단 공백을 줄이고 싶음",
            "market": "국내 동물병원 시장",
            "customer": "1~2인 소규모 동물병원, 야간진료 특화 동물병원",
        },
        {
            "industry": "기타",
            "problem": "블록체인 기반으로 여러 병원의 의료정보를 안전하게 유통하는 플랫폼",
            "market": "국내 의료정보 유통 시장",
            "customer": "종합병원 IT 부서",
        },
    ]

    for s in samples:
        print("=" * 60)
        print("입력:", s)
        result = extract_idea_requirements(s)
        print(json.dumps(result, ensure_ascii=False, indent=2))
