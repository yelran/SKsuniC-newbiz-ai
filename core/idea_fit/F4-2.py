import os

from pydantic import BaseModel, Field

from f4_common import (
    build_candidates_text,
    build_org_profile_text,
    filter_evidence_ids,
    get_openai_client,
    valid_evidence_ids,
)

OPENAI_MODEL = os.environ.get("F4_2_OPENAI_MODEL", "gpt-5.6-terra")
MAX_CANDIDATES_TO_LLM = 6  # 프롬프트 토큰(=비용) 통제용 상한


# ---------------------------------------------------------------
# 1. OpenAI structured output 스키마
# ---------------------------------------------------------------
class RankedCandidateLLM(BaseModel):
    id: str = Field(..., description="입력으로 받은 후보 id 그대로. 새 id를 만들지 마라.")
    name: str
    recommendation_reason: str = Field(
        ..., description="2~3문장. 왜 이 순위인지, 조직의 어떤 역량이 유리/불리하게 작용하는지"
    )
    transferability_note: str = Field(..., description="기술 전이 가능성에 대한 1~2문장 코멘트")
    cited_evidence_ids: list[str] = Field(
        default_factory=list,
        description="근거로 인용한 evidence_id. 조직 역량 프로필에 실제 존재하는 것만.",
    )


class MatchJudgmentLLMOutput(BaseModel):
    ranked_candidates: list[RankedCandidateLLM]
    overall_summary: str = Field(..., description="전체 후보군에 대한 2~4문장 총평")


# ---------------------------------------------------------------
# 2. 프롬프트
# ---------------------------------------------------------------
SYSTEM_PROMPT = (
    "너는 신사업 진입 전략 분석가다. 조직의 역량 프로필과 이미 점수가 계산된 신사업 후보 "
    "목록을 보고, 단순 total_score 순이 아니라 '기술 전이 가능성'(조직이 가진 역량을 이 "
    "사업에 실제로 얼마나 잘 옮겨 쓸 수 있는지) 관점까지 반영해 후보를 재정렬하고, 각 "
    "후보마다 추천 이유를 설명한다.\n\n"
    "규칙:\n"
    "1. 추천 이유에서 조직 역량을 근거로 들 때는 반드시 [조직 역량 프로필]에 실제로 있는 "
    "evidence_id를 인용해라. 없는 evidence_id를 지어내지 마라.\n"
    "2. 입력으로 받은 후보는 하나도 빠뜨리지 말고 전부 순위에 포함해라.\n"
    "3. 입력 목록에 없는 새 후보를 만들어내지 마라. 이 요청은 재정렬만 한다.\n"
)


# ---------------------------------------------------------------
# 3. 응답 검증/정리 (환각 방어)
# ---------------------------------------------------------------
def _validate_and_clean(ranked: list, sent_ids: set, valid_ev: set) -> tuple[list, list]:
    cleaned, dropped = [], []
    seen_ids = set()

    for rc in ranked:
        rc_dict = rc.model_dump()

        if rc.id in seen_ids:
            dropped.append({**rc_dict, "_drop_reason": f"중복 id({rc.id}) — 먼저 나온 항목만 유지"})
            continue
        if rc.id not in sent_ids:
            dropped.append({**rc_dict, "_drop_reason": "LLM에게 보여주지 않은 id (환각 의심)"})
            continue
        seen_ids.add(rc.id)

        kept_ev, bad_ev = filter_evidence_ids(rc.cited_evidence_ids, valid_ev)
        rc_dict["cited_evidence_ids"] = kept_ev
        if bad_ev:
            rc_dict["_dropped_evidence_ids"] = bad_ev

        rc_dict["source"] = "db"
        cleaned.append(rc_dict)

    for i, c in enumerate(cleaned, start=1):
        c["rank"] = i
    return cleaned, dropped


def _ensure_all_candidates_present(ranked: list, original: list, sent_ids: set) -> list:
    """LLM 응답에 없는 후보를 원 순서로 보충.

    프롬프트에 애초에 안 보낸 후보(MAX_CANDIDATES_TO_LLM 초과)와, 보냈는데 응답에서
    빠진 후보는 원인이 다르므로 사유 메시지를 구분한다.
    """
    present = {r["id"] for r in ranked}
    for c in original:
        cid = str(c.get("id"))
        if cid in present:
            continue
        reason = (
            "LLM 응답에서 누락되어 원본 순서를 유지합니다."
            if cid in sent_ids
            else "후보 수가 많아 이번 LLM 요청에 포함되지 않아 원본 순서를 유지합니다."
        )
        ranked.append({
            "id": cid,
            "name": c.get("name", ""),
            "source": "db",
            "recommendation_reason": reason,
            "transferability_note": "",
            "cited_evidence_ids": [],
            "rank": len(ranked) + 1,
        })
    return ranked


def _fallback_result(candidates: list, reason: str) -> dict:
    """API 키 없음/호출 실패 시: F3 점수 순서를 그대로 반환해 대시보드가 죽지 않게 한다."""
    ranked = [
        {
            "id": str(c.get("id")),
            "name": c.get("name", ""),
            "source": "db",
            "recommendation_reason": "LLM 매칭 판단을 사용할 수 없어 F3 점수 순서를 그대로 사용합니다.",
            "transferability_note": "",
            "cited_evidence_ids": [],
            "rank": i + 1,
        }
        for i, c in enumerate(candidates)
    ]
    return {
        "ranked_candidates": ranked,
        "overall_summary": "",
        "used_llm": False,
        "error": reason,
        "dropped_invalid_entries": [],
    }


# ---------------------------------------------------------------
# 4. 메인 함수
# ---------------------------------------------------------------
def llm_match_judgment(org_profile: dict, candidates: list) -> dict:
    """
    Args:
        org_profile: F1 산출물 (organization_profile.json 구조)
        candidates:  [{"id","name","total_score","scores":{...8개},"matched":[...],"missing":[...]}, ...]

    Returns:
        {
          "ranked_candidates": [
              {"id","name","source":"db","rank","recommendation_reason",
               "transferability_note","cited_evidence_ids"}
          ],
          "overall_summary": str,
          "used_llm": bool,
          "error": str | None,
          "dropped_invalid_entries": [...],   # 검증 실패로 제외된 원본 LLM 출력(디버그용)
        }
    """
    if not candidates:
        return {
            "ranked_candidates": [], "overall_summary": "", "used_llm": False,
            "error": "candidates가 비어 있습니다.", "dropped_invalid_entries": [],
        }

    valid_ev = valid_evidence_ids(org_profile)
    sent_ids = {str(c.get("id")) for c in candidates[:MAX_CANDIDATES_TO_LLM]}

    try:
        client = get_openai_client()
    except RuntimeError as e:
        return _fallback_result(candidates, reason=str(e))

    user_prompt = (
        f"[조직 역량 프로필]\n{build_org_profile_text(org_profile)}\n\n"
        f"[후보 목록 — F3 점수 계산 완료됨]\n"
        f"{build_candidates_text(candidates, MAX_CANDIDATES_TO_LLM)}"
    )

    try:
        resp = client.responses.parse(
            model=OPENAI_MODEL,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            text_format=MatchJudgmentLLMOutput,
        )
        parsed = resp.output_parsed
    except Exception as e:
        return _fallback_result(candidates, reason=f"OpenAI 호출 실패: {e}")

    ranked, dropped = _validate_and_clean(parsed.ranked_candidates, sent_ids, valid_ev)
    ranked = _ensure_all_candidates_present(ranked, candidates, sent_ids)

    return {
        "ranked_candidates": ranked,
        "overall_summary": parsed.overall_summary,
        "used_llm": True,
        "error": None,
        "dropped_invalid_entries": dropped,
    }


if __name__ == "__main__":
    import json

    # API 키 없이 실행하면 폴백 경로만 확인된다(비용 0).
    demo_org_profile = {
        "evidence_ids": ["ev_patent_001"],
        "capabilities": [{
            "capability_id": "cap_001",
            "name": "의료·수의 영상 AI(진단/검출)",
            "level": 5,
            "evidence_ids": ["ev_patent_001"],
        }],
        "evidence": {"ev_patent_001": {"type": "patent", "title": "X-ray 판독 보조 장치"}},
        "org_context": {"has_domain_expert": True, "patent_count": 35},
    }
    demo_candidates = [
        {"id": "idea_042", "name": "산업용 비파괴검사 AI 자동평가", "total_score": 63.2,
         "scores": {"조직역량적합도": 13.0}, "matched": ["cap_001", "cap_003"], "missing": []},
    ]
    print(json.dumps(llm_match_judgment(demo_org_profile, demo_candidates),
                     ensure_ascii=False, indent=2))
