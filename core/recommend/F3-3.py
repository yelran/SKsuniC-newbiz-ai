"""
F3-3. 실행 가능성 필터 (8단계)

확정 배점표(2026-08-09)에 맞춰 임계값을 비율 기준으로 판정한다.

■ 왜 비율 기준인가
  이전 코드는 절대값이었다: `진입장벽 <= 4`, `부족역량수준 < 5`.
  확정 배점에서 부족역량수준은 15점 만점이므로 기존 5점 허들은 약 33%다.
  배점이 다시 바뀌어도 의미가 유지되도록 만점 대비 비율로 판정한다.

■ 미확인(None) 처리
  점수가 None인 항목은 '자료 없음'이지 '나쁨'이 아니다. None인 항목으로는 탈락시키지
  않는다. 이전 코드는 None과 int를 비교해 TypeError로 죽었다(F3-4 통합 시 발견).

filter_feasibility(candidates, constraints) -> 통과 후보 리스트
"""

# 항목별 만점 (F3-2 조직계열 55점 + F2-5 시장계열 45점 = 100점)
CAPS = {
    "조직역량적합도": 20,
    "역량전이가능성": 15,
    "부족역량수준": 15,
    "실행가능성": 5,
    "시장성": 15,
    "경쟁강도": 10,
    "진입장벽": 10,
    "사업성": 10,
}

DEFAULTS = {
    "min_total_score": 40.0,    # 100점 환산 총점 하한
    "min_barrier_ratio": 0.40,  # 진입장벽 만점 대비 (10점 만점 → 4점)
    "min_gap_ratio": 1 / 3,     # 부족역량수준 만점 대비 (15점 만점 → 5점)
}


def _ratio(scores: dict, key: str):
    """만점 대비 비율. 값이 없거나 None이면 None(판정 보류)."""
    v = scores.get(key)
    if v is None:
        return None
    cap = CAPS.get(key)
    return v / cap if cap else None


def filter_feasibility(candidates: list, constraints: dict = None) -> list:
    """제약을 통과한 후보만 돌려준다. 각 후보에 feasible / drop_reason을 기록한다.

    탈락 조건
      1. 총점이 min_total_score 미만
      2. 진입장벽이 낮고(=장벽이 높고) 부족역량도 큰 최악의 조합
         — 둘 중 하나만 나빠서는 탈락시키지 않는다.
    """
    c = {**DEFAULTS, **(constraints or {})}
    passed = []

    for cand in candidates:
        scores = cand.get("scores", {}) or {}
        reasons = []

        total = cand.get("total_score")
        if total is not None and total < c["min_total_score"]:
            reasons.append(f"총점 {total} < {c['min_total_score']}")

        barrier = _ratio(scores, "진입장벽")
        gap = _ratio(scores, "부족역량수준")
        if (barrier is not None and gap is not None
                and barrier <= c["min_barrier_ratio"] and gap < c["min_gap_ratio"]):
            reasons.append(
                f"진입장벽 {scores['진입장벽']}/{CAPS['진입장벽']} + "
                f"부족역량 {scores['부족역량수준']}/{CAPS['부족역량수준']} 동시 미달")

        cand["feasible"] = not reasons
        cand["drop_reason"] = " / ".join(reasons) if reasons else None
        if cand["feasible"]:
            passed.append(cand)

    return passed


if __name__ == "__main__":
    samples = [
        {"name": "통과", "total_score": 85,
         "scores": {"진입장벽": 8, "부족역량수준": 12}},
        {"name": "총점 미달", "total_score": 35,
         "scores": {"진입장벽": 5, "부족역량수준": 8}},
        {"name": "장벽+역량 동시 미달", "total_score": 50,
         "scores": {"진입장벽": 3, "부족역량수준": 2}},
        {"name": "장벽만 높음 (통과해야 함)", "total_score": 50,
         "scores": {"진입장벽": 3, "부족역량수준": 14}},
        {"name": "시장데이터 없음 None (탈락시키면 안 됨)", "total_score": 60,
         "scores": {"진입장벽": None, "부족역량수준": 2}},
    ]

    passed = filter_feasibility(samples)
    print(f"{'후보':<38}{'결과':<6}사유")
    print("-" * 90)
    for s in samples:
        print(f"{s['name']:<38}{'통과' if s['feasible'] else '탈락':<6}"
              f"{s.get('drop_reason') or ''}")
    print(f"\n통과 {len(passed)}건 / 전체 {len(samples)}건")
