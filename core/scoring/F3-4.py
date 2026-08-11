"""
F3-4. 8개 항목 통합 및 최종 점수/추천 산출 (확정 배점표 2026-08-06 기준)

■ 배점 (합 100점)
  조직계열 55 (F3-2) : 조직역량적합도 20 · 역량전이가능성 15 · 부족역량수준 15 · 실행가능성 5
  시장계열 45 (F2-5) : 시장성 15 · 경쟁강도 10 · 진입장벽 10 · 사업성 10
  ※ 실제 배점은 f3_2.CAPS + f2_5.CAPS에서 읽어오며, 여기 숫자는 설명용이다.

■ 미확인(None) 처리 — F2-5·F3-2와 동일 규약
  파싱이 안 되거나 평가 대상이 없는 항목은 0점이 아니라 None이다.
  0점을 주면 '자료가 없다'가 '나쁘다'로 바뀐다. None은 분모에서 빼고 100점으로 환산한다.
  덕분에 시장 데이터가 없는 LLM 추천 후보도 DB 후보와 같은 척도로 비교된다.

■ 이전 버전에서 제거한 것
  의존 파일을 못 찾을 때 쓰던 Dummy 폴백을 없앴다. 폴백이 있으면 F2-5/F3-2가 없어도
  '옛 배점(20/15/15/5 + 경쟁강도)'으로 조용히 계산돼서 틀린 점수를 정상 결과로 착각한다.
  지금은 파일을 못 찾으면 즉시 FileNotFoundError로 멈춘다.

실행: python F3-4.py
"""

import importlib.util
import os
from pathlib import Path

import pandas as pd

BASE = Path(__file__).parent

# 통합 폴더에서 F3-4가 쓰는 파일은 폴더가 갈린다.
#   core/scoring/  F2-5, F3-2   ·   core/recommend/  F3-3
# 그래서 자기 폴더만 봐서는 F3-3을 못 찾는다.
SEARCH_DIRS = [
    Path(os.environ["SUNIC_CODE_DIR"]) if os.environ.get("SUNIC_CODE_DIR") else None,
    BASE,
    BASE.parent / "recommend",          # core/scoring → core/recommend
    BASE.parent / "app" / "scoring",
    Path.home() / "Desktop" / "app" / "core" / "scoring",
    Path.home() / "Desktop" / "app" / "core" / "recommend",
    Path.home() / "Desktop" / "app" / "scoring",
    Path.home() / "Desktop" / "A19" / "test_code" / "F2",
]


def _load(name: str, filename: str):
    """app 안에서 돌 때는 core/paths.py의 로더를 쓴다.

    paths.load_module은 sys.modules 캐시를 쓴다. main.py가 이미 F2-5를
    올려놨는데 여기서 또 로드하면 배점 상수가 두 벌 생겨서, CAPS를 고쳐도
    한쪽만 바뀌는 사고가 난다. 캐시를 공유해야 한 벌로 유지된다.
    """
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
f3_3 = _load("f3_3", "F3-3.py")


# ════════════════════════════════════════════════════════════
# 1. 배점 — F3-2/F2-5가 각자 들고 있는 값을 합쳐 만든다 (하드코딩 금지)
# ════════════════════════════════════════════════════════════
CAPS = {**f3_2.CAPS, **f2_5.CAPS}
TOTAL_CAP = sum(CAPS.values())

if TOTAL_CAP != 100:
    raise ValueError(
        f"배점 합계가 100이 아닙니다: {TOTAL_CAP}\n  {CAPS}\n"
        "  F3-2.CAPS(55) + F2-5.CAPS(45)가 맞는지 확인하세요.")


# ════════════════════════════════════════════════════════════
# 2. 단일 후보 채점
# ════════════════════════════════════════════════════════════
def calculate_score(required_caps_text, market_row: dict,
                    org_context: dict = None, return_detail: bool = False) -> dict:
    """DB 한 행을 받아 8개 항목 + 총점을 계산한다.

    Args:
        required_caps_text : DB '필요역량태그' 원문 (또는 표준역량ID 리스트)
        market_row         : DB 한 행 dict — '시장규모', '진입장벽수준' 등
        org_context        : F2-5가 쓰는 조직 플래그
                             (has_commercialization_experience 등)
        return_detail      : True면 매칭·세부항목까지 반환

    Returns:
        {"scores": {8개 항목: float|None},
         "total_score": float,   # 미확인 항목을 뺀 분모로 100점 환산
         "raw_score": float,     # 환산 전 획득 점수
         "denominator": int}     # 실제 적용된 배점 합
    """
    org_context = org_context or {}

    org_scores, org_detail = f3_2.calculate_org_series_scores(
        required_caps_text, org_context, return_detail=True)

    # F2-5는 파싱된 값을 받는다 — 원시 시장규모(억달러)와 진입장벽 등급.
    # (이전 버전의 parse_market_size_to_category / parse_company_case_count는
    #  F2-3에서 사분위 방식으로 바뀌고 경쟁강도가 삭제되면서 사라졌다.)
    market_data = {
        "market_size_usd": f2_5.parse_market_size_usd(market_row.get("시장규모")),
        "entry_barrier": f2_5.parse_entry_barrier_level(market_row.get("진입장벽수준")),
    }
    market_scores = f2_5.calculate_market_series_scores(market_data, org_context)

    scores = {**org_scores, **market_scores}
    raw, den = f2_5.sum_with_denominator(scores, CAPS)
    total = round(raw / den * 100, 1) if den else 0.0

    result = {"scores": scores, "total_score": total,
              "raw_score": raw, "denominator": den}
    if return_detail:
        result["detail"] = org_detail
    return result


# ════════════════════════════════════════════════════════════
# 3. 전체 DB 스캔 → Top-K
# ════════════════════════════════════════════════════════════
def _find_db() -> Path:
    for d in SEARCH_DIRS + [BASE.parent.parent / "data",       # app/data
                            Path.home() / "Desktop" / "app" / "data"]:
        if d is None:
            continue
        p = d / "신사업_DB.xlsx"
        if p.exists():
            return p
    raise FileNotFoundError("신사업_DB.xlsx를 찾을 수 없습니다.")


def score_all(org_context: dict = None, df: pd.DataFrame = None) -> list:
    """DB 전체를 채점해 후보 리스트로 돌려준다 (필터·정렬 전)."""
    if df is None:
        df = pd.read_excel(_find_db())

    out = []
    for _, row in df.iterrows():
        r = calculate_score(row.get("필요역량태그", ""), row.to_dict(),
                            org_context, return_detail=True)
        out.append({
            "id": row.get("아이디어ID", "unknown"),
            "name": row.get("아이디어명", "unknown"),
            "total_score": r["total_score"],
            "raw_score": r["raw_score"],
            "denominator": r["denominator"],
            "scores": r["scores"],
            "matched": r["detail"]["matched"],
            "missing": r["detail"]["missing"],
        })
    return out


def get_top_k_recommendations(org_context: dict = None, top_k: int = 3,
                              constraints: dict = None) -> list:
    """F3-3 필터를 통과한 후보를 총점순 Top-K로 반환. 대시보드용 API."""
    candidates = score_all(org_context)
    constraints = constraints or {"min_total_score": 40}
    passed = f3_3.filter_feasibility(candidates, constraints)
    passed.sort(key=lambda x: x["total_score"], reverse=True)
    return passed[:top_k]


# ════════════════════════════════════════════════════════════
# 4. 검증
# ════════════════════════════════════════════════════════════
MENTOR_CASE = "산업용 비파괴검사 AI 자동평가"
MENTOR_TARGET = 70


def _print_scorecard(title: str, result: dict):
    print(f"\n{'=' * 58}\n{title}\n{'=' * 58}")
    print(f"{'평가 항목':<16}{'배점':>5}{'획득':>9}")
    print("-" * 58)
    for k, cap in CAPS.items():
        v = result["scores"].get(k)
        print(f"{k:<16}{cap:>5}{('—' if v is None else f'{v:.1f}'):>9}")
    print("-" * 58)
    print(f"{'합계':<16}{result['denominator']:>5}{result['raw_score']:>9.1f}")
    print(f"{'100점 환산':<16}{'':>5}{result['total_score']:>9.1f}")


if __name__ == "__main__":
    print("배점: " + " / ".join(f"{k} {v}" for k, v in CAPS.items()))
    print(f"합계 {TOTAL_CAP}점\n")

    # F2-5 시장계열이 쓰는 조직 플래그 (F1 organization_profile.json 기준: 둘 다 True)
    org_context = {
        "has_commercialization_experience": True,
        "has_relevant_network": True,
    }

    df = pd.read_excel(_find_db())

    # [검증 1] 멘토 검증 케이스
    hit = df[df["아이디어명"].astype(str).str.contains(MENTOR_CASE, na=False)]
    if hit.empty:
        print(f"⚠️ 멘토 검증 케이스('{MENTOR_CASE}')를 DB에서 찾지 못했습니다.")
    else:
        row = hit.iloc[0]
        res = calculate_score(row.get("필요역량태그", ""), row.to_dict(), org_context)
        _print_scorecard(f"[검증 1] {row['아이디어ID']} {row['아이디어명']}", res)
        ok = res["total_score"] >= MENTOR_TARGET
        print(f"\n  {'✅' if ok else '❌'} {MENTOR_TARGET}점 기준 "
              f"{'통과' if ok else '미달'} ({res['total_score']}점)")

    # [검증 2] Top-3
    print(f"\n{'=' * 58}\n[검증 2] 전체 DB Top-3 추천 (F3-3 필터 적용)\n{'=' * 58}")
    for i, c in enumerate(get_top_k_recommendations(org_context, top_k=3), 1):
        print(f"{i}위. [{c['id']}] {c['name']} — {c['total_score']}점 "
              f"(획득 {c['raw_score']}/{c['denominator']})")

    # [검증 3] 분포
    print(f"\n{'=' * 58}\n[검증 3] 전체 50건 점수 분포\n{'=' * 58}")
    allc = score_all(org_context, df)
    s = pd.Series([c["total_score"] for c in allc])
    print(f"  평균 {s.mean():.1f} · 표준편차 {s.std():.1f} · "
          f"최소 {s.min():.1f} · 최대 {s.max():.1f}")
    print(f"  70점 이상 {(s >= 70).sum()}건 · 40점 미만 {(s < 40).sum()}건")
    dens = pd.Series([c["denominator"] for c in allc]).value_counts().to_dict()
    print(f"  적용 분모 분포: {dens}  (100 미만이면 미확인 항목이 있다는 뜻)")

    pd.DataFrame([{
        "아이디어ID": c["id"], "아이디어명": str(c["name"])[:22],
        **{k: c["scores"].get(k) for k in CAPS},
        "획득": c["raw_score"], "분모": c["denominator"], "총점": c["total_score"],
    } for c in allc]).to_csv("f3_4_scores.csv", index=False, encoding="utf-8-sig")
    print("\n저장 완료: f3_4_scores.csv")
