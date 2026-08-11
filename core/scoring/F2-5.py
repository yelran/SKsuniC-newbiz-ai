import re
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).parent

# ════════════════════════════════════════════════════════════
# 배점 
# ════════════════════════════════════════════════════════════
CAPS = {"시장성": 15, "경쟁강도": 10, "진입장벽": 10, "사업성": 10}

# 시장규모 사분위 경계 (억달러) 
MARKET_Q = (35.5, 119.9, 290.2)
MARKET_BANDS = (4, 8, 12, 15)          # 시장성 15점
VIABILITY_BANDS = (1.5, 3.0, 4.5, 6.0)  # 사업성 중 시장규모 6점

# 진입장벽 (5단계 정밀화)
ENTRY_BARRIER_TABLE = {"하": 5, "중하": 4, "중": 3, "중상": 2, "상": 1}
ENTRY_BARRIER_F = 5      # F 사업화 역량 보유 시
VIABILITY_F = 4          # 사업성 중 F 사업화 역량

# 경쟁강도 매트릭스 테이블
COMPETITION_MATRIX = {
    ("대", "상"): 10.0, ("대", "중상"): 9.0, ("대", "중"): 6.0, ("대", "중하"): 4.0, ("대", "하"): 2.0,
    ("중", "상"): 8.0, ("중", "중상"): 7.0, ("중", "중"): 5.0, ("중", "중하"): 3.0, ("중", "하"): 1.0,
    ("소", "상"): 6.0, ("소", "중상"): 5.0, ("소", "중"): 4.0, ("소", "중하"): 2.0, ("소", "하"): 0.0
}


# ════════════════════════════════════════════════════════════
# 1. 전처리 — 자유서술 텍스트를 값으로
# ════════════════════════════════════════════════════════════

def parse_market_size_usd(text) -> float:
    """시장규모 텍스트에서 금액을 억달러 단위로 뽑는다. 실패하면 None.

    '약 38억달러 (2025년 …) → 2034년 91억달러 전망' 처럼 여러 숫자가 있으면
    최댓값을 쓴다(기존 동작 유지). 미래 전망이 섞이는 것은 알려진 한계다.
    """
    if pd.isna(text):
        return None
    s = str(text)
    for pat, mul in ((r"([\d,]+\.?\d*)\s*억달러", 1.0),
                     (r"([\d,]+\.?\d*)\s*조원", 1e12 / 1400 / 1e8),
                     (r"([\d,]+\.?\d*)\s*억원", 1e8 / 1400 / 1e8)):
        m = re.findall(pat, s)
        if m:
            return max(float(v.replace(",", "")) for v in m) * mul
    return None


def parse_entry_barrier_level(text) -> str:
    """진입장벽수준에서 상/중상/중/중하/하 5단계를 뽑는다. 실패하면 None."""
    if pd.isna(text):
        return None
    t = str(text).split("—")[0].strip()
    if "상" in t and "중" in t: return "중상"
    if "하" in t and "중" in t: return "중하"
    if "상" in t: return "상"
    if "하" in t: return "하"
    if "중" in t: return "중"
    return None


# ════════════════════════════════════════════════════════════
# 2. 점수 계산
# ════════════════════════════════════════════════════════════

def _band(usd: float, bands: tuple) -> float:
    """사분위 경계로 구간 점수를 매긴다."""
    q1, q2, q3 = MARKET_Q
    if usd < q1:
        return bands[0]
    if usd < q2:
        return bands[1]
    if usd < q3:
        return bands[2]
    return bands[3]


def score_market_size(market_size_usd: float) -> float:
    """시장성 — 최대 15점. 사분위 4구간. 미확인이면 None."""
    if market_size_usd is None:
        return None
    return _band(market_size_usd, MARKET_BANDS)


def score_entry_barrier(barrier_level: str,
                        has_commercialization_experience: bool = False) -> float:
    
    db_part = ENTRY_BARRIER_TABLE.get(barrier_level)
    if db_part is None:
        return None
    f_part = ENTRY_BARRIER_F if has_commercialization_experience else 0
    return float(db_part + f_part)


def score_business_viability(market_size_usd: float,
                             has_commercialization_experience: bool = False) -> float:
    
    if market_size_usd is None:
        return None
    f_part = VIABILITY_F if has_commercialization_experience else 0
    return round(_band(market_size_usd, VIABILITY_BANDS) + f_part, 1)

def _get_market_size_category(usd: float) -> str:
    """억달러(USD)를 기반으로 매트릭스용 대/중/소를 구분합니다."""
    if usd >= 7.14: return "대"       # 약 1조원 이상
    if usd >= 0.0714: return "중"     # 약 100억원 이상
    return "소"

def score_competitive_intensity(market_size_usd: float, barrier_level: str) -> float:
    """경쟁강도 — 최대 10점 (시장규모 × 진입장벽 매트릭스 활용)."""
    if market_size_usd is None or barrier_level is None:
        return None
    size_cat = _get_market_size_category(market_size_usd)
    return COMPETITION_MATRIX.get((size_cat, barrier_level), None)

def calculate_market_series_scores(market_data: dict, org_context: dict = None) -> dict:
    """시장계열 4개 항목을 계산한다. 값이 None이면 '계산 불가'다."""
    org_context = org_context or {}
    f = bool(org_context.get("has_commercialization_experience", False))
    usd = market_data.get("market_size_usd")
    barrier = market_data.get("entry_barrier")
    
    return {
        "시장성": score_market_size(usd),
        "경쟁강도": score_competitive_intensity(usd, barrier),
        "진입장벽": score_entry_barrier(barrier, f),
        "사업성": score_business_viability(usd, f),
    }



def sum_with_denominator(scores: dict, caps: dict = None) -> tuple:
    
    caps = caps or CAPS
    got = sum(v for v in scores.values() if v is not None)
    den = sum(caps[k] for k, v in scores.items() if v is not None and k in caps)
    return round(got, 1), den


# ════════════════════════════════════════════════════════════
# 3. 검증
# ════════════════════════════════════════════════════════════

def check_배점초과(all_scores: list):
    print("\n[체크 1] 배점 초과 — 자동")
    ok = True
    for label, cap in CAPS.items():
        over = [s for s in all_scores if s[label] is not None and s[label] > cap]
        if over:
            ok = False
            print(f"   {label}: {len(over)}건이 {cap}점 초과 — "
                  f"{[s['아이디어ID'] for s in over]}")
    print("  전항목 배점 이내" if ok else "  → 로직 재확인 필요")


def check_단조성():
    print("\n[체크 2] 시장규모가 커질수록 점수가 올라가는지 — 자동")
    ok = True
    prev = None
    for usd in [1, 30, 50, 100, 150, 250, 300, 3000]:
        s = score_market_size(usd)
        if prev is not None and s < prev:
            ok = False
        print(f"  {usd:>5}억달러 → 시장성 {s:>2}점 · 사업성(F없음) "
              f"{score_business_viability(usd)}점")
        prev = s
    print("  단조 증가" if ok else "  역전 구간 있음")

    print("\n  진입장벽 (F 사업화 역량 없음 / 있음)")
    for lv in ("상", "중", "하"):
        print(f"    {lv} → {score_entry_barrier(lv, False)}점 / "
              f"{score_entry_barrier(lv, True)}점")
    print("  ※ F 보너스 5점이 세 등급에 균등하게 붙는지 확인 (상한 버그 재발 방지)")


def check_복합등급(df: pd.DataFrame):
    print("\n[체크 3] 진입장벽 복합등급 파싱 — 자동")
    n_fail, n_compound = 0, 0
    for _, r in df.iterrows():
        raw = str(r["진입장벽수준"])
        head = raw.split("—")[0].strip().split("-")[0].strip()
        lv = parse_entry_barrier_level(r["진입장벽수준"])
        if lv is None:
            n_fail += 1
            print(f"   {r['아이디어ID']}: '{head}' 파싱 실패 → None")
        elif head not in ("상", "중", "하"):
            n_compound += 1
            print(f"   {r['아이디어ID']}: 복합등급 '{head}' → '{lv}' "
                  f"({ENTRY_BARRIER_TABLE[lv]}점)")
    print(f"  복합등급 {n_compound}건 처리 · 파싱 실패 {n_fail}건")


def check_quartile_drift(df: pd.DataFrame):
    print("\n[체크 4] 사분위 경계가 현재 DB와 맞는지 — 자동")
    vals = sorted(v for v in (parse_market_size_usd(x) for x in df["시장규모"])
                  if v is not None)
    if not vals:
        print("  파싱된 값이 없다.")
        return
    q = pd.Series(vals).quantile([.25, .5, .75]).round(1).tolist()
    print(f"  하드코딩 : {list(MARKET_Q)}")
    print(f"  현재 DB  : {q}  ({len(vals)}/{len(df)}건 파싱)")
    drift = [abs(a - b) / b for a, b in zip(q, MARKET_Q)]
    if max(drift) > 0.10:
        print(f"   최대 {max(drift):.0%} 어긋남 — MARKET_Q를 갱신할 것")
    else:
        print(f"   최대 {max(drift):.0%} 차이 — 갱신 불필요")


def check_변별력(all_scores: list):
    print("\n[체크 5] 항목별 변별력 — 자동")
    sc = pd.DataFrame(all_scores)
    print(f"  {'항목':<7}{'배점':>4}{'고유값':>7}{'표준편차':>9}{'최빈비율':>9}{'미확인':>7}")
    print("  " + "-" * 46)
    for label, cap in CAPS.items():
        col = sc[label].dropna()
        top = col.value_counts(normalize=True).max() if len(col) else 0
        print(f"  {label:<7}{cap:>4}{col.nunique():>7}{col.std():>9.2f}"
              f"{top * 100:>8.0f}%{sc[label].isna().sum():>7}")

    print("\n  항목 간 상관계수")
    corr = sc[list(CAPS)].corr().round(3)
    print("  " + " " * 8 + "".join(f"{c:>10}" for c in corr.columns))
    for i in corr.index:
        line = f"  {i:<8}"
        for j in corr.columns:
            r = corr.loc[i, j]
            line += f"{r:>9.3f}{'*' if i != j and abs(r) >= 0.9 else ' '}"
        print(line)
    print("  ※ 사업성↔시장성이 높은 것은 F2-3에서 확인된 알려진 한계다(배점 유지).")


# ════════════════════════════════════════════════════════════
# 4. 전체 실행
# ════════════════════════════════════════════════════════════

def run_all(org_context: dict = None) -> list:
    df = pd.read_excel(next(BASE.glob("*DB*.xlsx")))
    results = []
    for _, row in df.iterrows():
        market_data = {
            "market_size_usd": parse_market_size_usd(row["시장규모"]),
            "entry_barrier": parse_entry_barrier_level(row["진입장벽수준"]),
        }
        scores = calculate_market_series_scores(market_data, org_context)
        got, den = sum_with_denominator(scores)
        results.append({
            "아이디어ID": row["아이디어ID"],
            "아이디어명": row["아이디어명"],
            **scores,
            "소계": got,
            "배점합": den,
            "환산(%)": round(got / den * 100, 1) if den else None,
        })
    return results


if __name__ == "__main__":
   
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    df = pd.read_excel(next(BASE.glob("*DB*.xlsx")))

    
    ORG = {"has_commercialization_experience": True}

    all_scores = run_all(ORG)

    out = pd.DataFrame(all_scores)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 150)
    print(f"배점: {CAPS} · 합계 {sum(CAPS.values())}점 (경쟁강도 10점은 조직계열로 이동)")
    print(f"조직 컨텍스트: {ORG}\n")
    print(out.to_string(index=False))

    check_배점초과(all_scores)
    check_단조성()
    check_복합등급(df)
    check_quartile_drift(df)
    check_변별력(all_scores)

    out.to_csv(BASE / "f2_5_scores.csv", index=False, encoding="utf-8-sig")
    print(f"\n저장: f2_5_scores.csv")
    print("\n[남은 확인] F3-2 조직계열과 합쳐 멘토님 케이스 2개를 검증할 것 "
          "(F3-4가 현재 Dummy로 돌고 있어 팀원에게 전달 필요)")
