"""
F4-3. DB 밖 신규 후보 제안 (6B단계, 담당: 김민주)

신사업 DB 50건에 없는 사업 아이디어를, 조직 역량 프로필(F1)만 보고 LLM이 제안한다.
F2-2(DB 내 검색)가 못 잡는 영역을 메우는 용도.

원래 F4-2.py의 `allow_new_candidates=True` 서브 기능이었으나 별도 파일로 분리했다.
분리 이유:
  - 재정렬(F4-2)은 후보가 있으면 항상 돌지만, 신규 제안은 켜고 끄는 선택 기능이라
    호출 시점과 비용 단위가 다르다.
  - 한 프롬프트에서 둘을 같이 시키면 신규 제안이 기존 후보 순위에 섞여 들어가
    "이건 DB 후보인가 지어낸 건가"를 사후 검증으로 갈라내야 해서 복잡해진다.

환각 방어(이중 방어)
--------------------
1. 스키마 단계: required_capability_ids는 표준역량 32개(f4_common.CAP_ID_VALUES) 중
   에서만 응답 가능(Literal enum). (2026-08-09: 조직 보유 11개 축이었으나, 그러면
   신규 제안도 조직이 이미 가진 역량만 요구하는 걸로 나와 7A 재채점 시 역량전이
   가능성이 항상 0점으로 깔리는 문제가 있어 32개로 넓혔다.)
2. 사후 검증 단계:
   - capability_id 화이트리스트 재확인 (enum을 통과해도 한 번 더)
   - cited_evidence_ids는 org_profile에 실제 있는 것만 통과
   - required_capability_ids가 비면 드롭 (근거 없는 제안 차단)
   - 기존 후보와 이름이 겹치면 드롭 (DB에 이미 있는 걸 "신규"라고 우기는 경우)
   - max_new 초과분은 잘라냄

에러 처리: OPENAI_API_KEY가 없거나 호출이 실패하면 빈 리스트를 반환한다
(used_llm=False, error에 사유). 신규 제안은 없어도 대시보드가 돌아가야 하는 부가 기능이다.

propose_new_candidates(org_profile, existing_candidates=None, max_new=2) -> JSON
"""

import os

from pydantic import BaseModel, Field

from f4_common import (
    CAP_NAME_BY_ID,
    CAPABILITY_MENU,
    CapabilityIdLiteral,
    build_candidates_text,
    build_org_profile_text,
    filter_evidence_ids,
    get_openai_client,
    valid_evidence_ids,
)

OPENAI_MODEL = os.environ.get("F4_3_OPENAI_MODEL", "gpt-5.6-sol")
# DB 전체와 안 겹치게 하려면 프롬프트에도 전체를 보여줘야 한다 — 일부만 보여주면
# LLM이 화면/한도 밖의 나머지와 겹치는 아이디어를 "신규"라고 제안할 수 있다.
# 고정 개수로 자르면 DB가 그 개수보다 커졌을 때 같은 문제가 조용히 재발하므로,
# 상한을 두지 않고 항상 existing_candidates 전체 길이를 쓴다(호출부에서 결정).
DEFAULT_MAX_NEW = 2


# ---------------------------------------------------------------
# 1. OpenAI structured output 스키마
# ---------------------------------------------------------------
class NewCandidateLLM(BaseModel):
    name: str = Field(..., description="신규 사업 아이디어 이름 (20자 내외)")
    description: str = Field(..., description="어떤 사업인지 2~3문장 설명")
    # F4-1(5A)의 입력 4필드와 같은 축 — "이 아이디어로 적합도 판단하기"를 누르면
    # 아이디어 입력 폼(진출산업/목표시장/해결문제/목표고객)에 그대로 채워 넣는다.
    # description은 "해결 문제" 칸에 들어가므로 여기 3개만 추가로 필요하다.
    target_industry: str = Field(..., description="진출 산업 (예: 수의·반려동물 헬스케어)")
    target_market: str = Field(..., description="목표 시장 (예: 국내 동물병원 시장)")
    target_customer: str = Field(..., description="목표 고객 (예: 1~2인 소규모 동물병원)")
    recommendation_reason: str = Field(
        ..., description="조직의 어떤 역량 때문에 이 사업이 가능하다고 보는지 2~3문장"
    )
    transferability_note: str = Field(..., description="기술 전이 가능성 1~2문장")
    required_capability_ids: list[CapabilityIdLiteral] = Field(
        ..., description="이 아이디어에 필요한 역량 카테고리. 목록에 있는 값만 사용."
    )
    cited_evidence_ids: list[str] = Field(
        default_factory=list,
        description="근거로 인용한 evidence_id. 조직 역량 프로필에 실제 존재하는 것만.",
    )


class NewCandidatesLLMOutput(BaseModel):
    new_candidates: list[NewCandidateLLM]
    overall_summary: str = Field(..., description="왜 이 방향들을 제안하는지 2~3문장 총평")


# ---------------------------------------------------------------
# 2. 프롬프트
# ---------------------------------------------------------------
def _build_system_prompt(max_new: int) -> str:
    return (
        "너는 신사업 진입 전략 분석가다. 조직의 역량 프로필을 보고, 이미 검토 중인 후보 "
        "목록에는 없지만 이 조직이라면 시도해볼 만한 신규 사업 아이디어를 제안한다.\n\n"
        "규칙:\n"
        f"1. 최대 {max_new}개까지만 제안해라. 억지로 채우지 말고, 근거가 약하면 적게 내라.\n"
        "2. [이미 검토 중인 후보]와 같거나 사실상 같은 아이디어는 제안하지 마라.\n"
        "2-1. target_industry(진출 산업)·target_market(목표 시장)·target_customer(목표 고객)도 "
        "구체적으로 채워라 — 사용자가 이 제안을 '아이디어 적합도 판단' 화면에 그대로 가져가서 "
        "쓸 수 있어야 한다.\n"
        "3. 각 제안마다 required_capability_ids를 반드시 채워야 하며, 그 값은 아래 "
        "[표준역량 카테고리 목록]에 있는 ID 중에서만 골라야 한다(목록에 없는 새 "
        "카테고리를 만들어내면 안 된다). 이 목록은 조직이 이미 보유한 역량이 아니라 "
        "신사업 판단에 쓰이는 표준 분류 전체다 — 이 아이디어를 실행하는 데 실제로 "
        "필요한 역량을 고르되, 조직이 지금 갖고 있는지 여부는 신경 쓰지 마라(보유 "
        "여부는 이후 단계가 별도로 판정한다).\n"
        "4. 조직 역량을 근거로 들 때는 [조직 역량 프로필]에 실제로 있는 evidence_id를 "
        "인용해라. 없는 evidence_id를 지어내지 마라.\n\n"
        f"[표준역량 카테고리 목록]\n{CAPABILITY_MENU}\n"
    )


# ---------------------------------------------------------------
# 3. 응답 검증/정리 (환각 방어)
# ---------------------------------------------------------------
def _normalize(name: str) -> str:
    return "".join(name.split()).lower()


def _validate_and_clean(proposed: list, valid_ev: set, existing_names: set,
                        max_new: int) -> tuple[list, list]:
    cleaned, dropped = [], []
    seen_names = set()

    for nc in proposed:
        d = nc.model_dump()
        key = _normalize(nc.name)

        if not nc.required_capability_ids:
            dropped.append({**d, "_drop_reason": "required_capability_ids가 비어 있음"})
            continue
        # enum을 통과했어도 한 번 더 화이트리스트 확인 (이중 방어)
        bad_caps = [c for c in nc.required_capability_ids if c not in CAP_NAME_BY_ID]
        if bad_caps:
            dropped.append({**d, "_drop_reason": f"목록에 없는 capability_id: {bad_caps}"})
            continue
        if key in existing_names:
            dropped.append({**d, "_drop_reason": "이미 검토 중인 후보와 중복"})
            continue
        if key in seen_names:
            dropped.append({**d, "_drop_reason": "제안 내 중복"})
            continue
        if len(cleaned) >= max_new:
            dropped.append({**d, "_drop_reason": f"max_new({max_new}) 초과"})
            continue
        seen_names.add(key)

        kept_ev, bad_ev = filter_evidence_ids(nc.cited_evidence_ids, valid_ev)
        d["cited_evidence_ids"] = kept_ev
        if bad_ev:
            d["_dropped_evidence_ids"] = bad_ev

        d["id"] = f"new_{len(cleaned) + 1}"
        d["source"] = "llm_proposed"
        # F3-1/F3-2는 매칭 축으로 capability_id가 아니라 F1의 name 문자열을 쓰므로,
        # LLM은 enum 안전성이 높은 id로 답하게 하고 여기서 name으로 변환해 같이 실어준다.
        d["required_capability_names"] = [CAP_NAME_BY_ID[c] for c in nc.required_capability_ids]
        cleaned.append(d)

    return cleaned, dropped


def _empty_result(reason: str) -> dict:
    return {
        "new_candidates": [],
        "overall_summary": "",
        "used_llm": False,
        "error": reason,
        "dropped_invalid_entries": [],
    }


# ---------------------------------------------------------------
# 4. 메인 함수
# ---------------------------------------------------------------
def propose_new_candidates(org_profile: dict, existing_candidates: list = None,
                           max_new: int = DEFAULT_MAX_NEW) -> dict:
    """
    Args:
        org_profile:         F1 산출물 (organization_profile.json 구조)
        existing_candidates: 이미 검토 중인 후보 (중복 제안 방지용, 없어도 됨)
        max_new:             제안 최대 개수

    Returns:
        {
          "new_candidates": [
              {"id":"new_1","name","description","source":"llm_proposed",
               "target_industry","target_market","target_customer",  # 5A 입력폼 프리필용
               "recommendation_reason","transferability_note",
               "cited_evidence_ids", "required_capability_ids",
               "required_capability_names"}          # F3-1 투입용 name 축
          ],
          "overall_summary": str,
          "used_llm": bool,
          "error": str | None,
          "dropped_invalid_entries": [...],
        }
    """
    existing_candidates = existing_candidates or []

    if max_new <= 0:
        return _empty_result("max_new가 0 이하입니다.")
    if not org_profile.get("capabilities"):
        return _empty_result("조직 역량 프로필이 비어 있어 제안할 근거가 없습니다.")

    try:
        client = get_openai_client()
    except RuntimeError as e:
        return _empty_result(str(e))

    existing_text = (
        build_candidates_text(existing_candidates, len(existing_candidates))
        if existing_candidates else "(없음)"
    )
    user_prompt = (
        f"[조직 역량 프로필]\n{build_org_profile_text(org_profile)}\n\n"
        f"[이미 검토 중인 후보 — 이것과 겹치지 않게]\n{existing_text}"
    )

    try:
        resp = client.responses.parse(
            model=OPENAI_MODEL,
            input=[
                {"role": "system", "content": _build_system_prompt(max_new)},
                {"role": "user", "content": user_prompt},
            ],
            text_format=NewCandidatesLLMOutput,
        )
        parsed = resp.output_parsed
    except Exception as e:
        return _empty_result(f"OpenAI 호출 실패: {e}")

    existing_names = {_normalize(str(c.get("name", ""))) for c in existing_candidates}
    cleaned, dropped = _validate_and_clean(
        parsed.new_candidates, valid_evidence_ids(org_profile), existing_names, max_new
    )

    return {
        "new_candidates": cleaned,
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
        {"id": "idea_042", "name": "산업용 비파괴검사 AI 자동평가", "total_score": 63.2},
    ]
    print(json.dumps(propose_new_candidates(demo_org_profile, demo_candidates),
                     ensure_ascii=False, indent=2))
