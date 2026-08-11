"""
신사업 진단 AI —  대시보드 통합 (Streamlit)
================================================
인트로 → STEP1~3 파일 업로드 → 분석 완료 → 대시보드

■ 연결된 파트 (파이프라인 12단계 기준)
  1~4   F1     업로드 → 파싱 → 조직 역량 프로필 (세션 전용, 서버 저장 없음)
  4     F2     profile_schema 표준 스키마 검증
  5B·6B F2-2   DB 후보 검색  ·  F4-3  DB 밖 신규 후보 LLM 제안
  5A·6A F4-1   아이디어 요구역량 추출  ·  F4-2  LLM 매칭 판단
  7     F3-4   8개 항목 통합 100점 (= F3-2 조직계열 55 + F2-5 시장계열 45)
  8     F3-3   실행 가능성 필터
  9     F3-1   역량 교집합·차집합 (F3-2가 내부에서 사용)
  10~11 F5     갭 리포트 · 보완전략 · 로드맵
  12    F6     이 파일 (화면 라우팅·시각화)

■ 축 연결 방식
  · F1은 업로드된 조직 정보를 표준역량 축 CAP_* 32개에 매핑한다.
  · 아이디어 요구역량(F4-1)도 같은 32개 표준역량 축을 직접 반환한다. 따라서
    별도 번역 없이 required_capability_ids를 F4-4와 F3-2에 전달한다.
  · 목록 밖 자유서술 역량은 표준 점수와 분리해 AI 판단임을 화면에 표시한다.
  · 조직 사실(보유수준·특허건수·전담인력·로드맵연계)은 apply_org_levels()가
    업로드 프로필의 standard_capability_levels를 F3-2에 주입한다. 주입하지 않으면
    F3-2가 data/표준역량_정의.csv(샘플 조직 스냅샷)를 읽어, 누가 업로드해도
    조직계열 55점이 같아진다.
  · CSV에 남는 것은 조직과 무관한 정의뿐이다 — 표준역량명·역량성격·특허분류·전이출처.

■ 남은 격차
  - 로드맵연계의 '확장'·'보조' 등급은 업로드 양식에 신호가 없어 '핵심'으로 잡힌다
    (샘플 기준 15개 중 13개 일치). 배점 영향은 B로드맵연계 4점.
  - 전담인력은 특허 발명자 + staff_keyword로 판정해, CSV의 수동 배정보다 넓게 잡힌다
    (샘플 기준 4개 역량이 '없음 → 있음'). C도메인인력부재 8점이 다소 높게 나온다.
  - F2-4는 배점을 '설계'하는 분석 스크립트라 런타임에 부르지 않는다.
    그 결론(배점)은 F3-2.CAPS + F2-5.CAPS 상수로 들어와 있다.
  임의 값으로 채우지 않는다. 미평가 항목은 0이 아니라 None으로 두고 분모에서 뺀다.

실행:  streamlit run main.py
"""

import base64
import html
import math
import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# 경로는 core/paths.py에서만 정의한다. 여기서 Path를 새로 만들지 말 것.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.paths import (  # noqa: E402
    ASSETS_DIR, DATA_DIR, DB_PATH, LOGO_PATH, SAMPLES_DIR,
    IDEA_FIT_DIR, PROFILE_DIR, RECOMMEND_DIR, RESULT_DIR, SCORING_DIR,
    UPLOAD_DIR, load_module, try_load_module,
)

st.set_page_config(page_title="suniC · 신사업 진단 AI", page_icon="🩺", layout="wide")


# ── 근거 ID(ev_patent_001 · ev_staff_노다혜) 화면에서 감추기 ──────────
#   LLM에게 근거 ID 인용을 강제하는 이유는 지어낸 문장을 막기 위해서다
#   (F4-2/F4-3이 응답을 받은 뒤 화이트리스트로 실재하는 ID인지 대조한다).
#   그래서 ID 자체는 계속 받아야 하고, 사용자에게 보일 때만 지운다.
#   ⚠️ 원본 dict는 건드리지 않는다 — 검증·PDF·디버그가 그대로 쓴다.
_EV_ID = r"ev_[A-Za-z0-9_가-힣]+"
_EV_GROUP = re.compile(                      # (ev_a, ev_b) / [ev_a] / (예: ev_a)
    r"\s*[\(\[]\s*(?:예\s*[:：]\s*)?" + _EV_ID
    + r"(?:\s*[,、·]\s*" + _EV_ID + r")*\s*[\)\]]")
_EV_BARE = re.compile(r"\s*[,、·]?\s*" + _EV_ID)      # 괄호 밖 낱개
_EV_TIDY = [                                          # 지운 자리 뒷정리
    (re.compile(r"\(\s*\)|\[\s*\]"), ""),
    (re.compile(r"\s+([,.·、)\]])"), r"\1"),
    (re.compile(r"([(\[])\s+"), r"\1"),
    (re.compile(r"\s{2,}"), " "),
]


def strip_evidence_ids(text: str) -> str:
    """화면 표시용으로 근거 ID를 지운다. 없는 문장은 그대로 돌려준다."""
    if not text:
        return text
    out = _EV_GROUP.sub("", str(text))
    out = _EV_BARE.sub("", out)
    for pat, rep in _EV_TIDY:
        out = pat.sub(rep, out)
    return out.strip()


def _safe_md_text(text: str) -> str:
    """LLM 자유서술을 st.markdown / st.write 에 그대로 넘길 때 쓴다.

    Streamlit Markdown이 `$...$`를 수식으로, `~`를 취소선 문법으로 해석해서
    "$2.5M~80M 규모" 같은 문장이 통째로 깨진다. 마크다운 특수문자만 백슬래시로
    막는다 — html.escape()는 여기서 쓰면 안 된다. unsafe_allow_html=False인
    st.markdown은 HTML을 직접 이스케이프하므로, 미리 &amp;로 바꿔 넘기면
    화면에 '&amp;'라는 글자가 그대로 보인다.
    """
    return (text or "").replace("\\", "\\\\").replace("$", "\\$").replace("~", "\\~")


def _safe_llm_text(text: str) -> str:
    """LLM 자유서술을 직접 만든 HTML 문자열 안에 끼워 넣을 때 쓴다.

    unsafe_allow_html=True로 넘기는 f-string 전용이다. 마크다운 이스케이프에
    더해 html.escape()까지 적용한다. st.markdown/st.write에 문자열을 그대로
    넘기는 자리에는 _safe_md_text()를 쓸 것.
    """
    return html.escape(_safe_md_text(text))


def _brief(node: dict, key: str) -> str:
    """대시보드용 짧은 버전(F5의 *_brief)을 꺼낸다.

    F5가 같은 LLM 호출에서 원문과 짧은 버전을 함께 만든다. 화면은 짧은 쪽을,
    PDF는 원문을 쓴다. *_brief가 없는 예전 리포트에서는 원문으로 되돌린다.
    """
    return str(node.get(f"{key}_brief") or node.get(key) or "")


def _f5_detail_block(pairs) -> str:
    """'자세히 보기' 안을 '라벨 : 본문' 블록으로 그린다.

    문단만 이어 붙이면 어디까지가 원인이고 어디부터 영향인지 구분이 안 돼서
    라벨을 굵게 앞세운다. 길이는 F5가 이미 줄여 보내므로 여기서 자르지 않는다.
    """
    return "".join(
        f'<div class="f5-dt-row"><p class="f5-dt-label">{html.escape(str(label))}</p>'
        f'<p class="f5-dt-body">{_safe_llm_text(str(text))}</p></div>'
        for label, text in pairs if str(text or "").strip()
    )


def _download_filename_part(value: object, fallback: str = "gap_report") -> str:
    """Windows에서도 안전한 다운로드 파일명 조각을 만든다."""

    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", str(value or "").strip())
    cleaned = re.sub(r"\s+", "_", cleaned).strip("._ ")
    return cleaned[:60] or fallback


# ════════════════════════════════════════════════════════════
# 파트 모듈 로드
#   필수  : 없으면 앱이 뜨지 않는다
#   선택  : 없으면 해당 화면만 '연결 대기'로 표시한다
# ════════════════════════════════════════════════════════════
f2_5 = load_module("f2_5", "F2-5.py")
profile_schema = load_module("profile_schema", "profile_schema.py")

# 선택 — 없거나 로드 실패하면 해당 기능만 '연결 대기'로 표시하고 앱은 계속 뜬다.
#   ⚠️ F2-4는 여기서 로드하지 않는다. 엔트로피 가중치를 '설계'하는 분석 스크립트이고,
#      그 결론(배점)은 이미 F3-2.CAPS + F2-5.CAPS에 상수로 들어와 있다.
#      런타임에 또 부르면 같은 배점을 두 곳에서 정하게 된다.
f3_2 = try_load_module("f3_2", "F3-2.py")   # 조직계열 55점
f3_3 = try_load_module("f3_3", "F3-3.py")   # 실행 가능성 필터 (8단계)
f3_4 = try_load_module("f3_4", "F3-4.py")   # 8개 항목 통합 100점
f4_1 = try_load_module("f4_1", "F4-1.py")   # 아이디어 요구역량 추출
f4_2 = try_load_module("f4_2", "F4-2.py")   # LLM 매칭 판단·재정렬
f4_3 = try_load_module("f4_3", "F4-3.py")   # DB 밖 신규 후보 제안
f4_4 = try_load_module("f4_4", "F4-4.py")   # 7A: 아이디어 입력 전용 채점(LLM 시장 추정)
f5 = try_load_module("f5", "F5.py")         # 갭 리포트·보완 로드맵


@st.cache_data(show_spinner=False)
def _build_f5_pdf_download(report: dict, subject_name: str) -> bytes:
    """생성 완료된 리포트를 PDF로만 변환한다. 외부 API는 호출하지 않는다."""

    return f5.export_gap_report_pdf(report, subject_name=subject_name)

# 평가 항목 — (항목명, 배점, 담당 파트). F3-2.CAPS/F2-5.CAPS에서 그대로 읽어온다.
#   ⚠️ 예전엔 여기 (24/18/18/5 + 15/10/10)을 하드코딩해뒀었는데, F3-2/F2-5가
#      배점표를 (20/15/15/5 + 15/10/10/10, 경쟁강도 부활)로 바꾼 뒤에도 이 상수는
#      안 바뀌어서 화면에 경쟁강도 행이 아예 안 뜨고 나머지 배점도 틀리게
#      표시되고 있었다(2026-08-06 발견). 항목이 또 바뀌어도 이 파일을 안 고쳐도
#      되게, 실제 배점표(f3_2.CAPS/f2_5.CAPS)에서 매번 만든다.
def _build_items() -> list:
    org_caps = f3_2.CAPS if f3_2 else {
        "조직역량적합도": 20, "역량전이가능성": 15, "부족역량수준": 15, "실행가능성": 5}
    return ([(n, m, "F3") for n, m in org_caps.items()]
            + [(n, m, "F2") for n, m in f2_5.CAPS.items()])


ITEMS = _build_items()
MARKET_ITEMS = [(n, m) for n, m, part in ITEMS if part == "F2"]
MARKET_TOTAL = sum(m for _, m in MARKET_ITEMS)
ORG_TOTAL = sum(m for _, m, part in ITEMS if part == "F3")


# ════════════════════════════════════════════════════════════
# 데이터
# ════════════════════════════════════════════════════════════
def load_f1():
    return load_module("f1", "F1.py")


def _f1_version() -> float:
    """F1.py의 수정 시각. 캐시 키에 넣어 F1을 고치면 자동으로 다시 파싱하게 한다.

    ⚠️ F1은 importlib로 동적 로드하므로 Streamlit이 변경을 감지하지 못한다.
       이걸 안 넣으면 F1을 고쳐도 예전 프로필이 계속 나온다 (실제로 겪었다).
    """
    return (UPLOAD_DIR / "F1.py").stat().st_mtime


@st.cache_data(show_spinner="조직 데이터 분석 중…")
def build_profile_from_bytes(payload: dict, _version: float) -> dict:
    """업로드된 파일(bytes)로 F1을 실행한다.

    payload는 {"org_intro": bytes|None, "hr_info": bytes, "patent": bytes|None}.
    bytes를 키로 캐싱하므로 같은 파일을 다시 올리면 재파싱하지 않는다.
    """
    f1 = load_f1()
    return f1.build_organization_profile({
        "intro_pptx": payload.get("org_intro"),
        "staff_excel": payload.get("hr_info"),
        "patent_excel": payload.get("patent"),
    })


@st.cache_data(show_spinner="샘플 조직 데이터 분석 중…")
def build_profile_from_samples(_version: float) -> dict:
    """개발용 — samples/ 폴더가 있으면 그걸로 프로필을 만든다."""
    f1 = load_f1()
    pptx = next(SAMPLES_DIR.glob("*.pptx"), None)
    xlsx = next(SAMPLES_DIR.glob("*org_data*.xlsx"), None)
    if xlsx is None:
        raise FileNotFoundError("samples/에 조직 데이터 xlsx가 없다")
    return f1.build_organization_profile({
        "intro_pptx": str(pptx) if pptx else None,
        "staff_excel": str(xlsx),
    })


def _stamp(profile: dict, source: str) -> dict:
    """프로필에 출처·시각을 붙인다 (2-3 표준 스키마의 _meta)."""
    from datetime import datetime
    out = dict(profile)
    out["_meta"] = {
        "source": source,                                    # uploaded | samples | restored
        "files": dict(st.session_state.get("files", {})),
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    return out


def load_org_profile() -> dict:
    """조직 역량 프로필을 가져온다. 업로드한 파일만 쓴다.

    ⚠️ 파일에서 읽지 않는다. 배포 시 Streamlit Cloud는 컨테이너 하나를 모든
       사용자가 공유하므로, 고정 경로에 쓰고 읽으면 사용자 A가 올린 조직
       인력정보·특허목록이 사용자 B에게 그대로 보인다.
       st.session_state는 브라우저 세션마다 격리되므로 여기만 신뢰한다.
       → 창을 닫으면 사라진다.

    ⚠️ samples/를 자동으로 대체하지 않는다. 업로드가 없을 때 개발용 샘플로
       채우면 '아무것도 안 올렸는데 결과가 나오는' 상태가 된다.
       F3-4가 의존성이 없을 때 Dummy 클래스로 조용히 돌던 것과 같은 문제다.
       개발 중 샘플을 쓰려면 URL에 ?dev=1 을 붙인다 (화면에 표시된다).
    """
    if st.session_state.get("profile"):
        return st.session_state["profile"]

    payload = {k: v for k, v in st.session_state.get("uploaded", {}).items() if v}
    if payload.get("hr_info"):
        try:
            profile = _stamp(build_profile_from_bytes(payload, _f1_version()), "uploaded")
        except Exception as e:
            st.session_state["_f1_error"] = f"업로드 파일 분석 실패 — {e}"
            return None
        st.session_state["profile"] = profile
        return profile

    if st.session_state.get("dev_samples") and SAMPLES_DIR.exists():
        try:
            profile = _stamp(build_profile_from_samples(_f1_version()), "samples")
        except Exception as e:
            st.session_state["_f1_error"] = str(e)
            return None
        st.session_state["profile"] = profile
        return profile
    return None


def apply_org_levels(profile: dict | None) -> tuple:
    """업로드된 조직 프로필을 F3-2에 주입한다 (4단계 → 7단계 연결).

    F1이 표준역량별로 만들어 준 보유수준·특허건수·전담인력·로드맵연계를 넘긴다.

    ⚠️ 이걸 안 하면 F3-2가 data/표준역량_정의.csv를 읽는다. 그 CSV는 샘플 조직의
       스냅샷이라, 다른 조직이 업로드해도 조직계열 점수가 똑같이 나온다
       (업로드→파싱 결과가 점수에 반영되지 않는다).

    ⚠️ @st.cache_data 안에서 부르면 안 된다. 캐시 적중 시 함수 본문이 실행되지 않아
       주입이 건너뛰어진다. 매 rerun마다 캐시 밖에서 부르고, 반환한 튜플을
       load_scored_db의 캐시 키로 넘겨 조직이 바뀌면 점수도 다시 계산되게 한다.
    """
    if not f3_2:
        return ()
    caps = (profile or {}).get("standard_capability_levels") or {}
    f3_2.set_org_capabilities(caps or None)
    return tuple(sorted(
        (k, v.get("level"), v.get("patent_count"), v.get("roadmap"),
         len(v.get("staff") or []))
        for k, v in caps.items()))


@st.cache_data
def load_scored_db(org_context: dict, org_levels: tuple = ()) -> pd.DataFrame:
    """신사업 DB 50건 × 8개 항목 점수 (7단계 '매칭 결과에 점수 매기기').

    F3-4가 있으면 조직계열 55점(F3-2) + 시장계열 45점(F2-5)을 합쳐 100점으로 채점하고,
    없으면 시장계열 45점만 채운다(조직계열은 None → 화면에서 '미연결').

    ⚠️ None은 0이 아니다. F2-5·F3-2·F3-4가 공통으로 쓰는 규약으로, '자료 없음'을
       0점으로 바꾸면 '나쁨'이 된다. None인 항목은 분모에서 빠지고 100점으로 환산된다.
       그래서 시장 데이터가 없는 후보도 같은 척도로 비교된다.
    """
    df = pd.read_excel(DB_PATH)
    rows = []
    for _, row in df.iterrows():
        tags = str(row["필요역량태그"]) if pd.notna(row["필요역량태그"]) else ""
        market_data = {
            "market_size_usd": f2_5.parse_market_size_usd(row["시장규모"]),
            "entry_barrier": f2_5.parse_entry_barrier_level(row["진입장벽수준"]),
        }

        if f3_4:
            r = f3_4.calculate_score(tags, row.to_dict(), org_context, return_detail=True)
            scores, det = r["scores"], r["detail"]
            total, raw, den = r["total_score"], r["raw_score"], r["denominator"]
            matched, missing = det["matched"], det["missing"]
        else:
            scores = f2_5.calculate_market_series_scores(market_data, org_context)
            raw, den = f2_5.sum_with_denominator(scores)
            total = round(raw / den * 100, 1) if den else 0.0
            matched, missing = [], []

        rows.append({
            "아이디어ID": row["아이디어ID"],
            "아이디어명": row["아이디어명"],
            "설명": row["설명"],
            "산업분류": row["산업분류"],
            "필요역량태그": tags,
            "실제기업사례": row["실제기업사례"],
            "기업사례_요약": row["기업사례_요약"],
            "출처링크": row["출처링크"],
            # 원문 컬럼을 그대로 남긴다. F3-4.calculate_score가 '시장규모'·'진입장벽수준'
            # 원문을 다시 파싱하므로, 이름을 바꾸면 시장계열 3개 항목이 조용히 None이 된다.
            "시장규모": row["시장규모"],
            "진입장벽수준": row["진입장벽수준"],
            "시장규모_억달러": market_data["market_size_usd"],
            "진입장벽_등급": market_data["entry_barrier"],
            **{name: scores.get(name) for name, _, _ in ITEMS},
            "총점": total,          # 100점 환산 (미평가 항목은 분모에서 제외)
            "획득": raw,
            "배점합": den,
            "matched": matched,
            "missing": missing,
            "sub_scores": det.get("sub_scores", {}) if isinstance(det, dict) else {},
        })
    return pd.DataFrame(rows)


def feasibility_pass(db: pd.DataFrame, constraints: dict = None) -> pd.DataFrame:
    """8단계 실행 가능성 평가 — F3-3 필터를 통과한 후보만 남긴다.

    F3-3은 후보 dict 리스트를 받으므로 DataFrame ↔ dict 변환을 여기서 한다.
    F3-3이 없으면 필터를 건너뛴다(전량 통과).
    """
    if not f3_3 or db.empty:
        return db.assign(feasible=True, drop_reason=None)

    cands = [{
        "id": r["아이디어ID"],
        "total_score": r["총점"],
        "scores": {name: r[name] for name, _, _ in ITEMS},
    } for _, r in db.iterrows()]
    f3_3.filter_feasibility(cands, constraints)

    by_id = {c["id"]: c for c in cands}
    out = db.copy()
    out["feasible"] = [by_id[i]["feasible"] for i in out["아이디어ID"]]
    out["drop_reason"] = [by_id[i]["drop_reason"] for i in out["아이디어ID"]]
    return out


@st.cache_resource(show_spinner="임베딩 모델 로딩 중… (최초 1회)")
def load_embedder():
    """F2-2.py 와 동일 모델. 최초 1회만 로딩."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("jhgan/ko-sroberta-multitask")


@st.cache_data(show_spinner=False)
def build_doc_embeddings(texts: tuple):
    model = load_embedder()
    return model.encode(list(texts), convert_to_tensor=True)


KEYWORD_MATCH_BONUS = 0.5   # F2-2.py 와 동일
MIN_ABSOLUTE_SCORE = 0.3    # F2-2.py 와 동일


def retrieve_business_candidates(db: pd.DataFrame, query: str, top_k: int = 3,
                                 mode: str = "키워드") -> pd.DataFrame:
    """F2-2. 검색어가 없으면 총점 순으로 반환."""
    if not query.strip():
        return db.sort_values("총점", ascending=False).head(top_k).assign(검색근거="8개 항목 총점순")

    terms = [t.strip() for t in query.replace(",", " ").split() if t.strip()]

    if mode == "임베딩":
        from sentence_transformers import util
        model = load_embedder()
        passages = tuple(db["아이디어명"] + ". " + db["필요역량태그"])
        doc_emb = build_doc_embeddings(passages)
        sims = util.cos_sim(model.encode(query, convert_to_tensor=True), doc_emb)[0].tolist()

        hits = []
        for idx, sim in enumerate(sims):
            tag_text = db.iloc[idx]["필요역량태그"]
            hit_count = sum(1 for t in terms if t in tag_text)
            combined = sim + KEYWORD_MATCH_BONUS * hit_count
            if combined >= MIN_ABSOLUTE_SCORE:
                hits.append((idx, combined, f"유사도 {sim:.3f} · 키워드 {hit_count}개"))
    else:
        hits = []
        for idx in range(len(db)):
            row = db.iloc[idx]
            blob = f"{row['아이디어명']} {row['필요역량태그']} {row['설명']}"
            hit_count = sum(1 for t in terms if t in blob)
            if hit_count:
                hits.append((idx, hit_count, f"키워드 {hit_count}개 일치"))

    if not hits:
        return db.iloc[0:0].assign(검색근거="")

    hits.sort(key=lambda x: x[1], reverse=True)
    hits = hits[:top_k]
    out = db.iloc[[h[0] for h in hits]].copy()
    out["검색근거"] = [h[2] for h in hits]
    return out


# ════════════════════════════════════════════════════════════
# 디자인 — onboarding_final.html 과 동일한 CSS
# ════════════════════════════════════════════════════════════
def logo_data_uri() -> str:
    if not LOGO_PATH.exists():
        return ""
    return "data:image/png;base64," + base64.b64encode(LOGO_PATH.read_bytes()).decode()


SPARKLE = ("url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'"
           "%3E%3Cpath fill='%23fff' d='M50 0C53 27 73 47 100 50 73 53 53 73 50 100 47 73 27 53 0 50 "
           "27 47 47 27 50 0Z'/%3E%3C/svg%3E\")")


# 업로드 박스 안에 넣을 문서 아이콘 (CSS ::before 용 data URI)
DOC_ICON_URI = (
    "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='30' height='36' "
    "viewBox='0 0 30 36' fill='none'%3E"
    "%3Crect x='1' y='1' width='28' height='34' rx='4' stroke='%23159A93' stroke-width='2'/%3E"
    "%3Cline x1='8' y1='12' x2='20' y2='12' stroke='%23B7E3E0' stroke-width='2.5' stroke-linecap='round'/%3E"
    "%3Cline x1='8' y1='18' x2='20' y2='18' stroke='%23B7E3E0' stroke-width='2.5' stroke-linecap='round'/%3E"
    "%3Cline x1='8' y1='24' x2='16' y2='24' stroke='%23B7E3E0' stroke-width='2.5' stroke-linecap='round'/%3E"
    "%3C/svg%3E")


def inject_css(screen: int, upload_title: str = "", uploaded: bool = False):
    logo = logo_data_uri()
    card_mode = 1 <= screen <= 4          # 블록 컨테이너 자체를 흰 카드로
    intro_mode = screen == 0

    # 파일을 올린 뒤에는 점선 박스 안에 올린 파일만 보이도록 (아이콘·제목·Browse 숨김)
    uploaded_css = ""
    if uploaded:
        uploaded_css = """
[data-testid="stFileUploaderDropzone"]{display:none}
[data-testid="stFileUploader"]{
  border:1.5px dashed var(--accent);border-radius:14px;background:#FAFEFE;
  min-height:250px;padding:16px;display:flex;flex-direction:column;
}
[data-testid="stFileUploader"] label{display:none}
[data-testid="stFileUploaderFile"]{
  background:var(--surface);border:1px solid var(--border);border-radius:10px;
  padding:10px 12px;font-size:12.5px;color:var(--text);
}
[data-testid="stFileUploaderFileName"]{font-size:12.5px;color:var(--text);font-weight:500}
"""

    # 인트로 진입 애니메이션 — 아래에서 위로 순차 등장
    intro_anim_css = ""
    if intro_mode:
        intro_anim_css = (
            ".intro-h1,.intro-sub,.intro-card,.stButton"
            "{animation:riseIn .6s cubic-bezier(.22,.61,.36,1) both}"
            ".intro-h1{animation-delay:.05s}"
            ".intro-sub{animation-delay:.16s}"
            ".intro-card{animation-delay:.27s}"
            ".stButton{animation-delay:.40s}"
        )

    # STEP 1~4 공통 레이아웃: 카드는 화면 정중앙, 버튼 줄은 항상 카드 맨 아래
    card_layout_css = ""
    if card_mode:
        card_layout_css = (
            "section.stAppViewMain{display:flex;flex-direction:column;justify-content:center;min-height:100vh}"
            ".block-container{display:flex;flex-direction:column}"
            ".block-container>[data-testid='stVerticalBlockBorderWrapper'],"
            ".block-container>[data-testid='stVerticalBlockBorderWrapper']>div"
            "{flex:1;display:flex;flex-direction:column}"
            ".block-container [data-testid='stVerticalBlock']{flex:1}"
            ".block-container [data-testid='stVerticalBlock']>[data-testid='stHorizontalBlock']"
            "{margin-top:auto;flex:0 0 auto}"
            "[data-testid='stHorizontalBlock'] .stButton>button"
            "{height:40px;min-height:40px;width:140px;padding:0 16px}"
            "[data-testid='stHorizontalBlock'] [data-testid='column']:last-child .stButton"
            "{display:flex;justify-content:flex-end}"
        )
    if screen == 4:
        # 분석 완료 화면: 남는 공간을 점선 박스 위·아래로 균등 분배 (진행바·제목은 그대로 위 고정)
        card_layout_css += (
            ".block-container [data-testid='stVerticalBlock']>[data-testid='element-container']"
            ":nth-child(3){margin-top:auto;margin-bottom:auto}"
        )
    if screen == 5:
        # 대시보드: 로고는 헤더 안에 있으므로 우상단 고정 로고는 숨기고, 버튼은 작게
        # 컬럼 안에 중첩된 가로 블록 = 헤더의 탭 그룹 (대시보드에서 여기뿐)
        head = "[data-testid='column'] [data-testid='stHorizontalBlock']"
        # 아이디어 입력 폼 카드 = .form-card-marker를 '바로 아래 단계'에 가진 bordered container.
        # 예전엔 .form-badge로 찾았는데, 탭 선택 버튼을 넣으려고 배지를 st.columns()
        # 안에 넣으면서 배지가 카드 기준 3단계 밑이 아니라 더 깊이 들어가 매칭이
        # 깨졌다(대신 badge_col 자신의 작은 래퍼가 잘못 매칭됨 — 실측 확인, 2026-08-06).
        # 그래서 배지와 별개로, 카드 맨 앞에 항상 1단계 깊이로 남는 숨김 마커를 둔다.
        form_card = ("[data-testid='stVerticalBlockBorderWrapper']"
                     ":has(>div>[data-testid='stVerticalBlock']"
                     ">[data-testid='element-container'] .form-card-marker)")
        # LLM추천 후보(F4-3) 카드 = .newcand-marker를 가진 bordered container.
        # 버튼을 카드 안에 넣기 위해 st.container(border=True)로 감싼다.
        newcand_card = ("[data-testid='stVerticalBlockBorderWrapper']"
                        ":has(>div>[data-testid='stVerticalBlock']"
                        ">[data-testid='element-container'] .newcand-marker)")
        # 아이디어 탭 줄 = .idea-tabs-marker 바로 다음에 오는 가로 블록.
        # (표식 자체는 안 보이게 숨기고, 형제 선택자로 그 다음 줄만 집어낸다)
        idea_tabs_mark = "[data-testid='element-container']:has(.idea-tabs-marker)"
        idea_tabs = f"{idea_tabs_mark}+[data-testid='stHorizontalBlock']"
        # 스코어카드 = .scorecard-marker를 가진 bordered container.
        # 'DB 원문보기' 버튼을 카드 우상단에 절대배치하려고 컨테이너로 감쌌다.
        scorecard_card = ("[data-testid='stVerticalBlockBorderWrapper']"
                          ":has(>div>[data-testid='stVerticalBlock']"
                          ">[data-testid='element-container'] .scorecard-marker)")
        card_layout_css += (
            ".ob-logo{display:none}"
            ".stButton>button{height:34px;min-height:34px;padding:0 15px;font-size:13px;"
            "border-radius:8px;box-shadow:none;font-weight:500}"
            ".stButton>button[kind='primary']{font-weight:600;box-shadow:none}"
            "[data-testid='stVerticalBlock']{gap:1rem}"      # 모든 카드 사이 간격 16px 동일
            "[data-testid='stHorizontalBlock']{align-items:stretch}"
            # 헤더 탭 — 예시 대시보드의 알약 그룹 (오른쪽 끝 정렬, 선택된 것만 흰 카드)
            f"{head}"
            "{background:var(--surface-2);border-radius:10px;padding:4px;gap:4px;"
            "width:fit-content;max-width:100%;margin-left:auto;margin-top:8px;"
            # 자리가 정말 없으면 겹치는 대신 아랫줄로 접힌다 (최후의 안전망)
            "flex-wrap:wrap;justify-content:flex-end}"
            # ⚠️ Streamlit 컬럼은 폭이 비율(%)로 정해져서, 창을 줄이면 칸이 버튼
            #    글자보다 좁아진다. 버튼은 white-space:nowrap이라 안 줄고 칸 밖으로
            #    삐져나가 옆 버튼과 겹쳤다. 칸을 '글자 폭만큼'으로 바꿔서 원인을 없앤다.
            f"{head}>[data-testid='column']"
            "{flex:0 0 auto;width:auto;min-width:0}"
            f"{head} .stButton>button"
            # 글씨만 줄이고 좌우 여백으로 보정해 버튼 박스 크기는 유지
            "{height:30px;min-height:30px;border-radius:7px;white-space:nowrap;"
            "width:100%;padding:0 20px;font-size:10.5px}"
            # 창이 좁아질수록 여백·글씨를 단계적으로 줄여 한 줄을 최대한 유지한다
            f"@media (max-width:1200px){{{head} .stButton>button"
            "{padding:0 14px;font-size:10px}}"
            f"@media (max-width:992px){{{head} .stButton>button"
            "{padding:0 9px;font-size:9.5px}}"
            f"@media (max-width:768px){{{head}{{margin-left:0}}"
            f"{head} .stButton>button{{padding:0 7px;font-size:9px}}}}"
            # Streamlit이 마크다운 끝의 <p> 여백을 상쇄하려고 넣는 -16px 때문에
            # 커스텀 HTML 카드의 아래가 잘려 다음 카드와 겹친다 → 0으로 되돌리고
            # 카드 사이 간격은 세로 블록의 gap 하나로 통일한다.
            "[data-testid='stMarkdownContainer']{margin-bottom:0}"
            # 아이디어 입력 폼 — st.container(border=True)를 예시의 흰 카드로.
            # 직계 경로로 한정해야 바깥 래퍼까지 카드가 되지 않는다.
            f"{form_card}"
            "{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:18px}"
            f"{form_card} [data-testid='stVerticalBlock']{{gap:0.6rem}}"
            # 마커 자체는 화면에 안 보여야 하는데, display:none이어도 그걸 감싼
            # element-container는 자리(gap)를 차지한다 — newcand_card와 같은 이유로 숨김.
            f"{form_card} [data-testid='element-container']:has(.form-card-marker)"
            "{display:none}"
            # ── 아이디어 탭 줄 — 카드 위에 '붙는' 브라우저 탭 모양 ──
            # 묶음 자체는 배경도 테두리도 없다. 탭 한 칸 한 칸이 카드와 같은 흰
            # 배경을 갖고, 아래 테두리만 지워서 바로 밑 입력 카드와 한 장처럼
            # 이어 보이게 한다. 예전엔 묶음에 연한 민트 배경을 깔고 선택된 탭만
            # 흰색으로 띄웠는데, 그러면 탭 줄이 카드와 분리된 별도 상자로 읽혔다.
            f"{idea_tabs_mark}{{display:none}}"
            f"{idea_tabs}"
            # margin-bottom은 카드 사이 기본 간격(stVerticalBlock gap:1rem=16px)을
            # 그대로 상쇄하는 값이다. 이 둘이 어긋나면 탭과 카드가 다시 떨어진다.
            "{width:fit-content;margin-left:auto;margin-bottom:-16px;gap:2px;"
            "padding:0;background:transparent;border:none;"
            "flex-wrap:nowrap;position:relative;z-index:1}"
            # 컬럼 폭을 내용(=버튼 30px)에 맞춘다. 비율(%)로 두면 화면 폭에 따라
            # 탭 사이가 벌어지거나 겹친다 — 헤더 탭과 같은 이유.
            f"{idea_tabs}>[data-testid='column']"
            "{flex:0 0 auto;width:auto;min-width:0}"
            f"{idea_tabs} .stButton>button"
            "{width:30px;height:24px;min-height:24px;padding:0;"
            "border-radius:8px 8px 0 0;font-size:11px;font-weight:500;"
            "background:var(--surface);color:var(--text-3);"
            "border:1px solid var(--border);border-bottom:none;box-shadow:none}"
            f"{idea_tabs} .stButton>button:hover"
            "{background:var(--accent-bg);color:var(--accent)}"
            # 선택된 탭은 민트로 꽉 채운다 — 지금 몇 번 탭인지 한눈에 보이게.
            f"{idea_tabs} .stButton>button[kind='primary'],"
            f"{idea_tabs} .stButton>button[kind='primary']:hover"
            "{background:var(--accent);color:#FFFFFF;font-weight:700;"
            "border-color:var(--accent)}"
            # 맨 끝 '＋'는 탭이 아니라 추가 버튼 — 민트 글씨로 구분한다.
            f"{idea_tabs}>[data-testid='column']:last-child .stButton>button"
            "{color:var(--accent);font-weight:400;font-size:13px}"
            # 4칸을 다 채우기 전에는 '적합도 판단하기'가 비활성이다.
            # 기본 primary 색이 그대로면 눌리는 버튼처럼 보여서 회색으로 낮춘다.
            f"{form_card} .stButton>button:disabled,"
            f"{form_card} .stButton>button:disabled:hover"
            "{background:var(--surface-2);color:var(--text-3);border:1px solid var(--border);"
            "box-shadow:none;cursor:not-allowed}"
            "[data-testid='stWidgetLabel'] p{font-size:12px;color:var(--text-2);margin-bottom:2px}"
            ".stTextInput input{height:36px;border-radius:8px;font-size:13px}"
            ".stTextArea textarea{border-radius:8px;font-size:13px}"
            # 입력창 테두리 — Streamlit 기본은 배경과 같은 색이라 안 보인다
            "[data-baseweb='input'],[data-baseweb='textarea']"
            "{border-color:var(--border);background:var(--surface)}"
            "[data-baseweb='base-input'],[data-baseweb='input'] input,"
            "[data-baseweb='textarea'] textarea{background:var(--surface)}"
            ".stSelectbox div[data-baseweb='select']>div"
            "{min-height:36px;border-radius:8px;border-color:var(--border);"
            "background:var(--surface);font-size:13px}"
            f"{head} .stButton>button[kind='primary']"
            "{background:var(--surface);color:var(--text);font-weight:500;"
            "box-shadow:0 1px 3px rgba(20,60,50,.12)}"
            f"{head} .stButton>button[kind='secondary']"
            "{background:transparent;color:var(--text-2);box-shadow:none;font-weight:400}"
            f"{head} .stButton>button[kind='secondary']:hover{{color:var(--text);background:transparent}}"
            # LLM추천 후보 카드 — note-card와 같은 톤 + 버튼·글씨를 더 작게
            f"{newcand_card}"
            "{background:var(--surface);border:1px solid var(--border);"
            "border-radius:12px;padding:16px}"
            f"{newcand_card} .stButton>button{{height:28px;min-height:28px;"
            "padding:0 12px;font-size:11px;border-radius:7px}"
            # 후보 카드마다 붙는 '이 아이디어로 적합도 판단하기'(secondary)는
            # 카드 안의 보조 동작이라 더 작게. '다시 제안받기'(primary)는 그대로 둔다.
            f"{newcand_card} .stButton>button[kind='secondary']"
            "{height:22px;min-height:22px;padding:0 8px;font-size:9px;border-radius:6px}"
            # 마지막 요소 = '다시 제안받기'. 바로 위 후보 카드 버튼과 붙어 보여서
            # 다른 동작인 걸 알기 어려웠다 → 위쪽만 따로 띄운다.
            f"{newcand_card}>div>[data-testid='stVerticalBlock']"
            ">[data-testid='element-container']:last-child{margin-top:26px}"
            # 카드 위/아래 여백을 같게 만든다.
            #   (카드 위 ~ 'LLM추천 후보')  ==  (버튼 아래 ~ 카드 아래)  == padding 16px
            #   ① 표식 span의 element-container는 display:none이어도 자리(gap)를 차지한다
            #   ② 제목·설명 <p>의 기본 margin이 위쪽 여백을 키운다 → 0으로 지우고
            #      요소 간 간격은 세로 블록 gap 하나로만 준다
            f"{newcand_card} [data-testid='element-container']:has(.newcand-marker)"
            "{display:none}"
            f"{newcand_card} [data-testid='stVerticalBlock']{{gap:10px}}"
            f"{newcand_card} .section-title{{margin:0 0 4px}}"
            f"{newcand_card} p{{margin:0}}"
            f"{newcand_card} [data-testid='stMarkdownContainer']{{margin:0}}"
            # 순위 카드 — 카드 위에 투명 버튼을 겹쳐 카드 전체를 클릭 영역으로 만든다.
            # (<a>를 쓰면 페이지가 새로 로드돼 업로드한 프로필이 날아간다)
            "[data-testid='column']:has(.card) [data-testid='stVerticalBlock']"
            "{position:relative}"
            "[data-testid='column']:has(.card) [data-testid='element-container']"
            ":has([data-testid='stButton'])"
            "{position:absolute;inset:0;margin:0;z-index:2}"
            "[data-testid='column']:has(.card) [data-testid='stButton'],"
            "[data-testid='column']:has(.card) [data-testid='stButton']>button"
            "{width:100%;height:100%;min-height:0;padding:0;border:none;"
            "background:transparent;box-shadow:none;opacity:0;cursor:pointer}"
            # 스코어카드 — 제목 옆에 'DB 원문보기'를 나란히
            f"{scorecard_card}"
            "{background:var(--surface);border:1px solid var(--border);"
            "border-radius:12px;padding:16px}"
            # 표식 span의 element-container는 display:none이어도 자리를 차지한다
            f"{scorecard_card} [data-testid='element-container']:has(.scorecard-marker)"
            "{display:none}"
            # 세로 블록을 '줄바꿈되는 가로 행'으로 바꾼다.
            # 제목·버튼은 내용만큼만 차지해 한 줄에 나란히 서고,
            # 표(.sc-body)는 width:100%라서 다음 줄로 넘어간다.
            f"{scorecard_card}>div>[data-testid='stVerticalBlock']"
            "{flex-direction:row;flex-wrap:wrap;align-items:baseline;gap:0 10px}"
            # 제목 칸은 '줄어들 수 있게'(flex-shrink:1 + min-width:0) 둔다.
            #
            # ⚠️ 아래 규칙들은 자손 결합자(공백) + !important로 쓴다. 직계 자식 체인
            #    ('>div>[stVerticalBlock]>[element-container]')으로 적었더니 실제
            #    사용자 화면에서 부모에는 flex-direction:row가 걸렸는데도 자식 폭만
            #    행 전체로 남아 버튼이 아랫줄로 떨어졌다(실측, 2026-08-09).
            #    Streamlit이 element-container에 width:100%를 직접 주기 때문에,
            #    중간 래퍼가 하나만 끼어도 체인이 끊기며 100%로 되돌아간다.
            f"{scorecard_card} [data-testid='stVerticalBlock']"
            " [data-testid='element-container']"
            "{width:auto !important;flex:0 1 auto;min-width:0}"
            # ⚠️ 제목 칸은 flex-basis를 auto로 두되 버튼 자리를 미리 빼둔다.
            #    flex-wrap:wrap은 '줄이기'보다 '줄바꿈'을 먼저 하기 때문에, 기준 폭이
            #    글자 폭 그대로면 제목이 남은 자리보다 조금만 길어도 제목을 줄이는
            #    대신 버튼을 아랫줄로 보내버린다. 그 뒤 혼자 남은 제목이 행 전체로
            #    늘어난다(실측: 행 387px, 제목 290+간격10+버튼89=389 → 2px 초과로 깨짐).
            #    flex-basis를 0으로 만들면 줄바꿈은 막히지만 flex-grow가 필요해지고,
            #    그러면 제목 칸이 행 전체로 자라 버튼이 오른쪽 끝까지 밀린다
            #    (원하는 모양은 '제목 바로 옆'이다).
            f"{scorecard_card} [data-testid='element-container']:has(.sc-title)"
            "{flex:0 1 auto !important;max-width:calc(100% - 105px)}"
            # element-container에만 min-width:0을 줘선 부족하다. 그 안의 마크다운
            # 래퍼가 기본값(min-width:auto)이라 '제목 글자 폭'만큼 버티기 때문에,
            # 안쪽까지 0으로 풀어야 .sc-title의 ellipsis가 동작한다.
            f"{scorecard_card} [data-testid='stMarkdownContainer']"
            "{min-width:0}"
            # 'DB 원문보기' 버튼은 절대 줄이지 않는다
            f"{scorecard_card} [data-testid='element-container']:has(.stButton)"
            "{flex:0 0 auto !important}"
            # 표는 반대로 항상 통째로 다음 줄을 차지한다
            f"{scorecard_card} [data-testid='element-container']:has(.sc-body)"
            "{width:100% !important;flex:0 0 100%;margin-top:10px}"
            f"{scorecard_card} .sc-title{{font-size:14px;font-weight:600;margin:0;"
            "white-space:nowrap;overflow:hidden;text-overflow:ellipsis}"
            # 제목보다 작은 글씨 + 연한 회색. baseline 정렬이라 제목 글자선에 맞는다.
            f"{scorecard_card} .stButton>button{{height:auto;min-height:0;line-height:1.4;"
            "padding:0;border:none;background:transparent;box-shadow:none;"
            "color:var(--text-3);font-size:11px;font-weight:400}"
            f"{scorecard_card} .stButton>button:hover{{color:var(--accent);background:transparent}}"
        )

    # F5 갭 리포트 카드(=.f5-card-anchor를 가진 bordered container) 선택자.
    # 아래 CSS에서 여러 번 쓰여 길이가 부담스러워 변수로 뺐다.
    F5_CARD = ('[data-testid="stVerticalBlockBorderWrapper"]'
               ':has(> div > [data-testid="stVerticalBlock"]'
               ' > [data-testid="element-container"] .f5-card-anchor)')
    # 갭 리포트의 버튼 줄 = .f5-actions-anchor 표식 바로 다음에 오는 가로 블록.
    F5_ACTIONS = ('[data-testid="element-container"]:has(.f5-actions-anchor)'
                  '+[data-testid="stHorizontalBlock"]')

    st.markdown(f"""
<style>
:root{{
  --surface:#FFFFFF; --surface-2:#F0F9F8;
  --border:#DFEFED; --text:#14312D; --text-2:#6C8A85; --text-3:#9FB8B4;
  --accent:#159A93; --accent-2:#0E837D; --accent-bg:#E4F6F4; --accent-soft:#D5EDEB;
  --warn-bg:#FBEFE3; --warn:#A05A16;
}}

/* ── 공통 배경 (그라데이션 + 반짝이 2개) ── */
.stApp{{
  background:
    {SPARKLE} no-repeat left 9% top 13% / 86px,
    {SPARKLE} no-repeat right 9% bottom 15% / 44px,
    radial-gradient(120% 85% at 12% 0%, rgba(255,255,255,.55) 0%, rgba(255,255,255,0) 50%),
    linear-gradient(135deg,#ECF7F3 0%,#DDF0E9 42%,#C6E6DC 100%);
  background-attachment:fixed;
}}
/* 왼쪽 아래에서 오른쪽으로 크게 휘어 지나가는 밝은 면 */
.stApp::before{{
  content:"";position:fixed;z-index:0;pointer-events:none;
  left:-72%;top:24%;width:190%;height:0;padding-bottom:190%;border-radius:50%;
  background:linear-gradient(120deg,rgba(255,255,255,.62) 0%,rgba(255,255,255,.2) 55%,rgba(255,255,255,0) 100%);
}}
[data-testid="stHeader"]{{background:transparent}}
[data-testid="stToolbar"], [data-testid="stDecoration"]{{display:none}}
footer, #MainMenu{{visibility:hidden}}

html, body, [class*="css"]{{font-family:"Pretendard","Apple SD Gothic Neo","Segoe UI",sans-serif}}
body{{color:var(--text)}}

/* 로고 — 오른쪽 위 고정 (여백 많은 원본 PNG를 크롭해서 사용) */
.ob-logo{{
  position:fixed;z-index:1000;top:22px;right:34px;width:110px;height:28px;
  display:block;cursor:pointer;
  background-image:url("{logo}");background-repeat:no-repeat;
  background-size:203px auto;background-position:-47px -54px;
  mix-blend-mode:multiply;
}}

/* ── 블록 컨테이너 ── */
/* 온보딩 카드는 화면 정중앙에 오도록 부모를 flex로 */
{card_layout_css}
.block-container{{position:relative;z-index:1;
  {"max-width:520px;background:var(--surface);border-radius:18px;box-shadow:0 6px 24px rgba(20,60,50,.09);padding:36px 34px 30px;margin-top:0;margin-bottom:0;height:670px;"
   if card_mode else
   ("max-width:680px;min-height:100vh;display:flex;flex-direction:column;justify-content:center;padding-top:0;padding-bottom:0;"
    if intro_mode else "max-width:1400px;padding-top:2.2rem;")}
}}

/* ── 버튼 ── */
.stButton>button{{
  background:var(--accent);color:#fff;border:none;font-weight:600;font-family:inherit;
  {"width:100%;border-radius:14px;padding:15px;font-size:15px;box-shadow:0 6px 16px rgba(21,154,147,.26);"
   if intro_mode else "border-radius:18px;padding:9px 20px;font-size:13px;box-shadow:0 4px 12px rgba(21,154,147,.24);"}
}}
.stButton>button[kind="primary"]{{background:var(--accent);color:#fff;border:none}}
.stButton>button:hover, .stButton>button[kind="primary"]:hover{{background:var(--accent-2);color:#fff;border:none}}
.stButton>button:focus, .stButton>button[kind="primary"]:focus{{color:#fff;box-shadow:none;border:none}}
.stButton>button[kind="primary"]:active{{background:var(--accent-2);color:#fff}}
/* st.download_button은 .stButton이 아니라 stDownloadButton으로 렌더링돼서 위 규칙이
   안 걸린다 — 테마 기본 primary(빨강)로 나오던 걸 같은 민트로 맞춘다.
   ⚠️ 자손 결합자(공백)로 써야 한다. help=를 주면 버튼이 툴팁 래퍼로 한 번 더
      감싸여서 '>button'(직계 자식)으로는 안 잡힌다(Streamlit 1.38 실측). */
[data-testid="stDownloadButton"] button{{border-radius:18px;padding:9px 20px;font-size:13px;
  font-weight:600;font-family:inherit;border:none;background:var(--accent);color:#fff;
  box-shadow:0 4px 12px rgba(21,154,147,.24)}}
[data-testid="stDownloadButton"] button:hover,
[data-testid="stDownloadButton"] button:active{{background:var(--accent-2);color:#fff;border:none}}
[data-testid="stDownloadButton"] button:focus{{color:#fff;box-shadow:none;border:none}}
[data-testid="stDownloadButton"] button p{{color:#fff}}
/* 테두리를 border 대신 inset 그림자로 그려서, primary 버튼과 높이가 정확히 같아지게 함 */
.stButton>button[kind="secondary"]{{
  background:transparent;color:var(--text-2);font-weight:500;border:none;
  box-shadow:inset 0 0 0 1px var(--border);
}}
.stButton>button[kind="secondary"]:hover,
.stButton>button[kind="secondary"]:focus{{
  background:var(--surface-2);color:var(--text);box-shadow:inset 0 0 0 1px var(--border);
}}

/* ── 인트로 ── */
.intro-h1{{margin:0 0 14px;font-size:27px;line-height:1.42;font-weight:700;color:var(--text);letter-spacing:-.4px;text-align:center}}
.intro-sub{{margin:0 0 26px;font-size:14px;color:var(--text-2);text-align:center}}
.intro-sub b{{color:var(--accent);font-weight:700}}
.intro-card{{background:var(--surface);border-radius:18px;padding:10px 6px;box-shadow:0 6px 24px rgba(20,60,50,.09);margin-bottom:30px}}
{intro_anim_css}
.feat{{display:flex;align-items:center;gap:14px;padding:16px}}
.feat .ico{{flex:0 0 40px;width:40px;height:40px;border-radius:11px;background:var(--accent-bg);display:flex;align-items:center;justify-content:center}}
.feat .t1{{margin:0 0 4px;font-size:14.5px;font-weight:600;color:var(--text)}}
.feat .t2{{margin:0;font-size:12.5px;color:var(--text-3)}}

/* ── 온보딩 ── */
.ob-progress{{display:flex;gap:8px;margin-bottom:24px}}
.ob-progress span{{flex:1;height:6px;border-radius:4px;background:var(--accent-soft)}}
.ob-progress span.on{{background:var(--accent)}}
.ob-step{{font-size:13px;font-weight:600;color:var(--text);margin:0 0 16px}}
.ob-head{{text-align:center;margin-bottom:6px}}
.ob-title{{font-size:15px;font-weight:600;color:var(--text);margin:12px 0 0}}
/* 위는 붙이고(=Streamlit 기본 간격만), 아래 버튼과는 넉넉히 — HTML 예시와 동일 */
.ob-tip{{background:var(--surface-2);border-radius:12px;padding:14px 16px;margin:0 0 24px}}
.ob-tip-title{{font-size:12.5px;font-weight:600;color:var(--text);margin:0 0 8px}}
.ob-tip-title i{{font-style:normal;color:var(--accent)}}
.ob-tip ul{{list-style:none;margin:0;padding:0}}
.ob-tip li{{font-size:12px;color:var(--text-2);margin:0 0 6px;padding-left:12px;position:relative}}
.ob-tip li:before{{content:"\\2022";position:absolute;left:0}}
.ob-done-title{{font-size:20px;font-weight:700;color:var(--text);margin:0 0 4px;letter-spacing:-.3px}}
/* 점선 박스는 파일 3개를 딱 맞게 감싼다. 남는 공간은 박스 위·아래 여백으로 */
.ob-result-box{{border:1.5px dashed var(--accent);border-radius:14px;background:#FAFEFE;
  padding:16px;margin-bottom:28px}}
.ob-chip{{display:flex;align-items:center;gap:14px;background:var(--accent-bg);border-radius:12px;padding:18px 20px;margin-bottom:12px}}
.ob-chip:last-child{{margin-bottom:0}}
.ob-chip.off{{background:var(--surface-2)}}
.ob-check{{width:24px;height:24px;border-radius:50%;border:2px solid var(--accent);color:var(--accent);
  display:flex;align-items:center;justify-content:center;font-size:13px;flex-shrink:0}}
.ob-chip.off .ob-check{{border-color:var(--text-3);color:var(--text-3)}}
.ob-chip .cname{{font-size:13.5px;font-weight:600;margin:0;line-height:1.5;
  min-height:0;color:var(--text)}}   /* 후보 카드용 min-height:38px 무효화 */
.ob-chip .csub{{font-size:11.5px;color:var(--text-2);margin:0;line-height:1.45}}
.ob-done-note{{font-size:12.5px;color:var(--text-2);line-height:1.6;margin:0 0 4px}}
/* 업로드된 파일 표시 — st.file_uploader의 점선 박스와 같은 모양으로 직접 그린다.
   (위젯을 다시 그리면 화면 이동 후 상태가 비므로) */
.up-box{{background:#FAFEFE;border:1.5px dashed var(--accent);border-radius:14px;
  min-height:250px;padding:16px;display:flex;flex-direction:column;
  margin-bottom:16px}}   /* st.file_uploader 위젯과 같은 아래 여백 */
.up-file{{display:flex;align-items:center;gap:12px;background:var(--surface);
  border:1px solid var(--border);border-radius:10px;padding:14px 16px}}
.up-doc{{width:22px;height:26px;flex-shrink:0;
  background:url("{DOC_ICON_URI}") no-repeat center/contain;opacity:.75}}
.up-name{{font-size:13.5px;font-weight:600;color:var(--text);overflow:hidden;
  text-overflow:ellipsis;white-space:nowrap}}
/* 파일명 뒤 크기 → 남는 공간을 밀어내 ✕를 오른쪽 끝으로 보낸다 */
.up-size{{font-size:12.5px;color:var(--text-3);flex:1;white-space:nowrap}}
.up-x{{flex-shrink:0;width:26px;height:26px;border-radius:6px;text-decoration:none;
  display:flex;align-items:center;justify-content:center;
  font-size:17px;line-height:1;color:var(--text-2)}}
.up-x:hover{{background:var(--surface-2);color:var(--text)}}

/* 파일 업로더 — HTML 예시의 큰 점선 박스와 동일하게 (아이콘·제목을 박스 안으로) */
[data-testid="stFileUploaderDropzone"]{{
  background:#FAFEFE;border:1.5px dashed var(--accent);border-radius:14px;
  min-height:250px;padding:28px 16px;gap:0;
  display:flex;flex-direction:column;align-items:center;justify-content:center;
}}
/* Streamlit 기본 영문 안내·클라우드 아이콘 숨김 */
[data-testid="stFileUploaderDropzoneInstructions"]{{display:none}}
[data-testid="stFileUploaderDropzone"]::before{{
  content:"";display:block;order:1;width:30px;height:36px;margin-bottom:14px;
  background:url("{DOC_ICON_URI}") no-repeat center/contain;
}}
[data-testid="stFileUploaderDropzone"]::after{{
  content:"{upload_title}";order:2;margin-bottom:20px;
  font-size:15px;font-weight:600;color:var(--text);
}}
[data-testid="stFileUploaderDropzone"] button{{
  order:3;background:var(--accent-bg);color:var(--accent);border:none;font-weight:600;
  border-radius:10px;padding:7px 16px;font-size:13px;box-shadow:none;
}}
[data-testid="stFileUploaderDropzone"] button:hover{{background:var(--accent-soft);color:var(--accent-2)}}
[data-testid="stFileUploaderFile"]{{font-size:12px;color:var(--text-2)}}
{uploaded_css}

/* ── 대시보드 ── */
.top{{display:flex;justify-content:space-between;align-items:flex-end;margin-bottom:18px;
  border-bottom:1px solid var(--border);padding-bottom:16px}}
.top h1{{font-size:20px;font-weight:600;margin:0}}
.top p{{margin:4px 0 0;color:var(--text-2);font-size:13px}}
/* 대시보드 헤더 — 로고 + 타이틀 (예시 대시보드와 동일) */
.brand{{display:flex;align-items:center;gap:14px}}
.brand-logo{{flex:0 0 auto;width:104px;height:26px;display:block;
  background-image:url("{logo}");background-repeat:no-repeat;
  background-size:192px auto;background-position:-44px -51px;mix-blend-mode:multiply}}
/* Streamlit 기본 h1(큰 글씨 + 위아래 padding)을 덮어쓴다 */
.brand h1{{font-size:20px;font-weight:600;margin:0;padding:0;line-height:1.35;letter-spacing:-.2px}}
.brand p{{margin:4px 0 0;padding:0;color:var(--text-2);font-size:13px;line-height:1.4}}
.top-line{{border:none;border-top:1px solid var(--border);margin:14px 0 16px}}
.org-summary{{display:flex;justify-content:space-between;align-items:center;background:var(--surface);
  border:1px solid var(--border);border-radius:12px;padding:14px 20px;margin:0}}
.org-summary .oname{{font-weight:600;font-size:14px}}
.org-summary .osub{{font-size:12px;color:var(--text-2);margin-top:2px}}
/* F1 조직 역량 칩 */
.cap-row{{display:flex;flex-wrap:wrap;gap:6px;margin-top:12px}}
.cap-chip{{display:inline-flex;align-items:center;gap:6px;font-size:11px;
  background:var(--surface-2);border:1px solid var(--border);border-radius:6px;padding:3px 8px}}
.cap-chip b{{font-weight:600;color:var(--text)}}
.cap-chip i{{font-style:normal;color:var(--accent);letter-spacing:-1px}}
.cap-chip u{{text-decoration:none;color:var(--text-3);font-size:10px}}
.ctx-row{{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}}
.ctx-flag{{font-size:10.5px;border-radius:5px;padding:2px 7px}}
.ctx-flag.on{{color:var(--accent);background:var(--accent-bg)}}
.ctx-flag.off{{color:var(--text-3);background:var(--surface-2)}}
/* 후보 카드 — 예시 대시보드와 동일 비율 (카드 전체가 클릭 영역) */
.card{{display:block;text-decoration:none;color:inherit;cursor:pointer;
  background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:16px}}
.card:hover{{border-color:var(--accent-soft)}}
.card.pick{{border:1.5px solid var(--accent);background:var(--accent-bg)}}
.rank{{font-size:11px;color:var(--text-3);letter-spacing:.02em}}
.card.pick .rank{{color:var(--accent);font-weight:600}}
.cname{{font-weight:600;font-size:14px;margin:6px 0 10px;line-height:1.35;min-height:38px;color:var(--text)}}
/* 링크 안이라 파랗게 물드는 걸 막고 본문색으로 고정 */
.cscore{{font-size:20px;font-weight:600;line-height:1.2;color:var(--text)}}
.cmax{{font-size:12px;color:var(--text-3);font-weight:400}}
.cscore-row{{display:flex;align-items:center;gap:12px}}
.gauge{{width:52px;height:52px;border-radius:50%;flex-shrink:0;display:flex;align-items:center;
  justify-content:center;position:relative}}
.gauge::before{{content:"";position:absolute;inset:5px;background:var(--surface);border-radius:50%}}
.gauge span{{position:relative;font-size:12px;font-weight:600}}
.csub{{font-size:12px;color:var(--text-2)}}
/* 후보 카드의 '진입장벽 · 시장규모' 한 줄 — 12px이면 '$'가 줄바꿈된다 */
.cmeta{{font-size:10.5px;white-space:nowrap;letter-spacing:-.2px}}
.tag{{display:inline-block;font-size:11px;background:var(--accent-bg);color:var(--accent);
  padding:2px 8px;border-radius:6px;margin-top:8px}}
.card.pick .tag{{background:var(--surface)}}
.section-title{{font-size:14px;font-weight:600;margin:0 0 12px}}
.canvaswrap{{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:16px;margin:0}}
/* 나란히 놓인 두 카드의 높이를 맞춘다.
   ⚠️ [data-testid="column"]에 height:100%를 주면 안 된다. 컬럼은 flex 아이템이고
      부모(stHorizontalBlock)의 높이가 내용으로 정해지므로 100%가 auto로 풀리면서
      align-items:stretch가 무력화된다(실측: 583 vs 298로 어긋났다).
      컬럼은 stretch에 맡기고, 그 안쪽 래퍼들만 100%로 이어 높이를 카드까지 전달한다. */
/* 나란히 놓인 두 카드의 높이를 맞춘다. (Streamlit <p> 태그 래퍼 이슈 해결)
   ⚠️ [data-testid="column"]에 height:100%를 주면 안 된다. ... (중략) ... */
/* 나란히 놓인 두 카드의 높이를 맞춘다. (Streamlit <p> 태그 래퍼 이슈 해결)
   ⚠️ [data-testid="column"]에 height:100%를 주면 안 된다. ... (중략) ... */
[data-testid="stHorizontalBlock"]:has(.eq-card) [data-testid="column"]>div,
[data-testid="stHorizontalBlock"]:has(.eq-card) [data-testid="stVerticalBlockBorderWrapper"]>div,
[data-testid="stHorizontalBlock"]:has(.eq-card) [data-testid="stVerticalBlock"],
[data-testid="stHorizontalBlock"]:has(.eq-card) [data-testid="element-container"],
[data-testid="stHorizontalBlock"]:has(.eq-card) [data-testid="stMarkdown"],
[data-testid="stHorizontalBlock"]:has(.eq-card) [data-testid="stMarkdownContainer"]{{
  height:100%}}
.eq-card{{height:100%;display:flex;flex-direction:column;box-sizing:border-box}}
/* 레이더는 '점수 산출 근거' 카드 높이에 맞춘다.
   ⚠️ svg를 일반 흐름에 두면 viewBox 비율(1:1) 때문에 폭만큼 높이를 차지해서
      레이더가 카드 높이를 끌어올린다(실측: 515px → 카드 583px).
      absolute로 흐름에서 빼면 레이더는 높이를 만들지 않고, 카드 높이는 옆
      카드에서 stretch로 정해진 뒤 그 안을 레이더가 채운다. */
/* 절대 위치(absolute)를 해제하고 최소 높이를 강제 고정 */
.radar-wrap{{flex:1; min-height:250px; display:flex; align-items:center; justify-content:center; margin-top:10px;}}
.radar-wrap svg{{width:100%; max-height:250px; display:block;}}
.sc-table{{width:100%;border-collapse:collapse;font-size:13px;border:none}}
.sc-table th{{text-align:left;font-weight:500;color:var(--text-3);font-size:11px;letter-spacing:.03em;
  padding:0 0 8px;border:none;border-bottom:1px solid var(--border);background:transparent}}
.sc-table th:not(:first-child), .sc-table td:not(:first-child){{text-align:center}}
.sc-table td{{padding:9px 0;border:none;border-bottom:1px solid var(--border);background:transparent}}
/* Streamlit이 tr에 넣는 위쪽 테두리 제거 (평가 항목 위 선) */
.sc-table tr{{background:transparent;border:none}}
.sc-table tr:last-child td{{border-bottom:none}}
.sc-table tr.wait td{{color:var(--text-3)}}
.score-cell{{font-weight:600}}
.bar-track{{background:var(--surface-2);border-radius:4px;height:6px;overflow:hidden;margin-top:5px}}
.bar-fill{{background:var(--accent);height:100%;border-radius:4px}}
.note-card{{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:16px;height:100%;margin:0}}
/* 아이디어 적합도 판단형 — 입력 폼 */
.form-badge{{display:inline-block;font-size:11px;color:var(--accent);background:var(--accent-bg);
  padding:3px 9px;border-radius:6px;margin-bottom:2px}}
.form-hint{{font-size:11px;color:var(--text-3);margin:6px 0 0}}
.stTextArea textarea, .stTextInput input{{font-size:13px}}
.field-label{{font-size:12px;color:var(--text-2);margin:0 0 4px}}
/* 인트로 등장 애니메이션 */
@keyframes riseIn{{from{{opacity:0;transform:translateY(26px)}}to{{opacity:1;transform:none}}}}
.note-card.wait{{background:var(--surface-2);border-style:dashed}}
.note-card p{{margin:0 0 6px;color:var(--text-2);font-size:13px}}
.note-card b{{color:var(--text);font-weight:600}}
/* DB 원문 — 라벨/값 2열. 값이 여러 줄로 흘러도 라벨 칸을 침범하지 않는다
   (전에는 한 문단이라 '실제기업사례' 라벨 아래로 둘째 줄이 내려왔다) */
.db-grid{{display:grid;grid-template-columns:max-content 1fr;
  column-gap:14px;row-gap:7px;align-items:start}}
.db-k{{font-size:13px;font-weight:600;color:var(--text);white-space:nowrap}}
.db-v{{font-size:13px;color:var(--text-2);line-height:1.6}}
/* DB 원문보기 팝업 — 닫기 X를 모서리 쪽으로 (기본값 top:30px/right:24px은 본문에 붙어 보인다) */
[data-testid="stDialog"] button[aria-label="Close"]{{top:12px;right:12px}}
.evidence-tag{{display:inline-block;font-size:9.5px;line-height:1.5;color:var(--accent);
  background:var(--accent-bg);padding:0 5px;border-radius:4px;margin-left:5px;vertical-align:middle}}
/* F5 갭 리포트 — 흰 카드와 작은 상태 포인트만 사용하는 절제된 정보 구조 */
/* 리포트 제목 — h3(28px)은 카드보다 과하게 커서 화면을 눌렀다. */
.f5-report-title{{font-size:21px;font-weight:500;color:var(--text);
  line-height:1.35;margin:0 0 14px;letter-spacing:-.01em}}
/* 탭 안 섹션 제목 — 4개 탭이 h5/굵은글씨로 제각각이라 크기가 달라 보였다.
   전부 이 클래스 하나로 통일한다. margin-top은 탭 줄과 제목을 띄우고,
   바로 아래 .f5-section-lead는 margin 0으로 제목에 붙인다. */
.f5-tab-title{{font-size:17px;font-weight:700;color:var(--text);
  line-height:1.4;margin:20px 0 0}}
/* 3개 요약 카드 — 왼쪽 민트 띠 + 번호 배지로 통일(색은 3개 모두 같은 민트). */
.f5-kpi{{position:relative;box-sizing:border-box;background:#FFFFFF;border:1px solid #D6E3DF;
  border-left:4px solid #2C7A73;border-radius:12px;padding:16px 18px;box-shadow:none}}
/* 카드 사이 '›' — 컬럼 간격(16px) 정중앙에 놓는다.
   right:-8px로 카드 오른쪽 8px 지점에 오른변을 두고, translateX(50%)로 자기 폭의
   절반만큼 밀어 중심이 정확히 간격 한가운데(=카드끝+8px)에 오게 한다.
   컬럼과 그 안쪽 래퍼 모두 overflow:visible이라 잘리지 않는다(실측). */
.f5-kpi-arrow{{position:absolute;right:-8px;top:50%;transform:translate(50%,-50%);
  color:#AFC4C0;font-size:17px;line-height:1;pointer-events:none;user-select:none}}
/* 폭이 좁아지면 Streamlit이 컬럼을 세로로 쌓는다 — 그때 화살표는 방향이 안 맞으니 숨긴다. */
@media (max-width:640px){{.f5-kpi-arrow{{display:none}}}}
.f5-kpi-head{{display:flex;align-items:center;gap:7px;margin:0 0 9px}}
.f5-kpi-num{{flex:0 0 auto;display:inline-flex;align-items:center;justify-content:center;
  width:21px;height:21px;border-radius:50%;background:#2C7A73;color:#FFFFFF;
  font-size:11px;font-weight:700;line-height:1}}
.f5-kpi-label{{font-size:11px;color:var(--text-2);margin:0;letter-spacing:.01em}}
.f5-kpi-value{{font-size:18px;font-weight:700;color:var(--text);line-height:1.35;margin:0 0 6px}}
.f5-kpi-sub{{font-size:11.5px;color:var(--text-2);line-height:1.5;margin:0}}
.f5-judgment{{background:#F4FBFA;border:1px solid #DFEFEC;
  border-radius:10px;padding:15px 17px;margin:12px 0 4px}}
.f5-judgment-label{{font-size:11px;font-weight:700;color:#2C7A73;margin:0 0 5px}}
.f5-judgment-text{{font-size:13.5px;font-weight:500;color:var(--text);line-height:1.7;margin:0}}
.f5-compact-card{{min-height:116px;box-sizing:border-box;background:#FFFFFF;border:1px solid #D6E3DF;
  border-left:4px solid #2C7A73;border-radius:12px;padding:15px;margin:0 0 8px}}
.f5-card-title{{font-size:13.5px;font-weight:700;color:var(--text);line-height:1.45;margin:0 0 7px}}
.f5-card-copy{{font-size:12px;color:#536B68;line-height:1.65;margin:8px 0 0}}
.f5-badge-row{{display:flex;flex-wrap:wrap;gap:5px;align-items:center}}
.f5-badge{{display:inline-block;font-size:10px;font-weight:650;line-height:1.45;padding:3px 8px;
  border-radius:999px;border:1px solid transparent;white-space:nowrap}}
.f5-strategy-build{{color:#19756F;background:#E5F6F3;border-color:#C9ECE7}}
.f5-strategy-buy{{color:#A35A0B;background:#FFF2DE;border-color:#F6DFC0}}
.f5-strategy-partner{{color:#3567B7;background:#EAF1FD;border-color:#D4E2FA}}
.f5-strategy-hire{{color:#AD4848;background:#FCECEC;border-color:#F4D5D5}}
.f5-status-recommended{{color:#16775A;background:#E5F6EE;border-color:#C9EAD9}}
.f5-status-conditional{{color:#956317;background:#FFF4DC;border-color:#F1DFC0}}
.f5-status-not_applicable{{color:#758581;background:#F1F5F4;border-color:#E1E9E7}}
.f5-priority-high{{color:#A7463F;background:#FCEDEA;border-color:#F4D5D1}}
.f5-priority-medium{{color:#956317;background:#FFF4DC;border-color:#F1DFC0}}
.f5-priority-low{{color:#3567B7;background:#EAF1FD;border-color:#D4E2FA}}
.f5-priority-maintain{{color:#16775A;background:#E5F6EE;border-color:#C9EAD9}}
.f5-priority-unscored{{color:#758581;background:#F1F5F4;border-color:#E1E9E7}}
/* 제목과 설명은 서로 다른 element-container라 사이에 세로 블록 gap(1rem=16px)이
   그대로 들어간다 — margin만 0으로 줄여선 안 붙는다. 음수 margin으로 gap을 상쇄한다. */
.f5-section-lead{{font-size:12px;color:var(--text-2);margin:-13px 0 14px;line-height:1.6}}
/* 나란히 놓인 F5 카드 — 접힌 상태에서는 높이가 같고 '+ 자세히 보기'도 같은 줄에
   오게 하되, 하나를 펼쳤을 때 옆 카드까지 같이 늘어나면 안 된다.
   ⚠️ align-items:stretch(기본)로 높이를 맞추면 행 높이가 '가장 큰 카드'로 정해져서
      한 장을 펼치는 순간 옆 카드도 같이 늘어난다. 그래서 stretch를 끄고(flex-start)
      대신 min-height로 접힌 높이를 맞춘다 — 펼친 카드만 그 아래로 자란다. */
[data-testid="stHorizontalBlock"]:has(.f5-eq-anchor){{align-items:flex-start}}
[data-testid="stHorizontalBlock"]:has(.f5-eq-anchor) [data-testid="stVerticalBlockBorderWrapper"]>div>[data-testid="stVerticalBlock"]{{
  display:flex;flex-direction:column;min-height:var(--f5-card-min,126px)}}
/* 링크를 바닥으로 밀어 카드마다 같은 높이에 오게 한다. */
[data-testid="stHorizontalBlock"]:has(.f5-eq-anchor) [data-testid="stExpander"]{{margin-top:auto}}
/* 카드 높이가 어긋나는 진짜 이유는 본문 줄 수 차이(1줄 vs 2줄)다. min-height만으로는
   본문이 긴 카드가 그 위로 자라 여전히 안 맞는다. 본문 칸을 2줄로 확보해 맞춘다
   (12px × line-height 1.65 × 2줄 = 39.6px). 3줄 이상은 그만큼만 더 길어진다. */
[data-testid="stHorizontalBlock"]:has(.f5-eq-anchor) .f5-card-copy{{min-height:40px}}
/* 로드맵은 3열이라 제목(headline)이 길어 접힌 높이가 조금 더 필요하다. */
[data-testid="stHorizontalBlock"]:has(.f5-roadmap-anchor) [data-testid="stVerticalBlockBorderWrapper"]>div>[data-testid="stVerticalBlock"]{{
  --f5-card-min:150px}}
/* 권장·최우선이 붙은 보완전략 카드는 '테두리 링'으로 강조한다.
   왼쪽 굵은 띠(다른 카드와 같은 모양)로는 구분이 안 돼서, 네 변을 같은 굵기의
   민트로 두르고 바깥에 옅은 링을 한 겹 더 얹는다. */
/* ⚠️ 테두리를 굵히면 그 카드만 2px 커져서 옆 카드와 높이가 어긋난다.
      굵은 선은 inset 그림자로 그려 레이아웃 크기를 그대로 둔다. */
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"] > [data-testid="element-container"] .f5-card-featured){{
  border-color:#2C7A73!important;border-left-width:1px!important;
  box-shadow:inset 0 0 0 2px #2C7A73, 0 0 0 3px rgba(44,122,115,.10)!important}}
/* PDF·전문 보기 두 버튼 — 크기를 정확히 같게 맞추고, 폭이 좁아져도 글자가
   두 줄로 접히지 않게 한다(접히면 한쪽만 키가 커져 짝이 안 맞았다).
   .stButton과 stDownloadButton은 기본 padding·font가 달라 각각 눌러 맞춘다. */
[data-testid="element-container"]:has(.f5-actions-anchor){{display:none}}
/* ⚠️ Streamlit 컬럼은 폭이 비율(%)이라 화면이 넓어질수록 버튼이 같이 늘어나
      둘 사이가 멀어졌다. 칸을 '내용 폭'으로 바꿔 버튼 크기만큼만 차지하게 한다
      (헤더 탭·아이디어 탭에서 쓴 것과 같은 방법). 남는 자리는 오른쪽에 남는다. */
/* 버튼 폭을 변수 하나로 묶어 '칸'과 '버튼'에 같이 먹인다.
   ⚠️ 칸을 width:auto(내용 폭)로 두면 안 된다. 안쪽 래퍼들이 %폭이라 내용 폭 계산에
      기여하지 않아 칸이 버튼보다 좁게 잡히고, 버튼이 칸 밖으로 삐져나와 옆 버튼과
      겹쳤다(넓은 화면에서만 드러남). 칸에도 같은 px을 직접 준다. */
{F5_ACTIONS}{{gap:8px;--f5-btn:208px;--f5-btn-fs:13px}}
@media (max-width:1200px){{{F5_ACTIONS}{{--f5-btn:186px;--f5-btn-fs:12px}}}}
@media (max-width:992px){{{F5_ACTIONS}{{--f5-btn:160px;--f5-btn-fs:11px}}}}
@media (max-width:768px){{{F5_ACTIONS}{{--f5-btn:138px;--f5-btn-fs:10px}}}}
/* var()가 어떤 이유로든 안 풀려도 칸이 0으로 주저앉지 않게 폴백값을 함께 준다.
   (폭이 정해지지 않으면 버튼이 칸 밖으로 나가 옆 버튼과 겹친다) */
{F5_ACTIONS}>[data-testid="column"]{{flex:0 0 auto;width:var(--f5-btn,208px);min-width:0}}
/* 마지막 빈 칸이 남는 자리를 모두 받아 두 버튼을 왼쪽에 붙여 둔다. */
{F5_ACTIONS}>[data-testid="column"]:last-child{{flex:1 1 auto;width:auto}}
/* download_button은 help= 때문에 툴팁 래퍼가 한 겹 더 있어 width:100%가 래퍼 폭을
   따라간다 — 버튼에도 같은 px을 직접 줘서 두 버튼 크기를 맞춘다. */
{F5_ACTIONS} .stButton>button,
{F5_ACTIONS} [data-testid="stDownloadButton"] button{{
  width:var(--f5-btn,208px);height:40px;min-height:40px;padding:0 10px;
  font-size:var(--f5-btn-fs,13px);white-space:nowrap;border-radius:18px}}
/* 펼친 내용은 '자세히 보기' 링크에서 한 칸 띄운다 — 바로 붙으면 링크와 본문이
   한 덩어리로 읽혔다. Streamlit expander의 본문은 details 바로 아래 div다. */
{F5_CARD} [data-testid="stExpander"] details>div{{padding-top:12px}}
/* '다시 생성'은 바로 위 카드에 붙어 보여서 한 칸 띄운다.
   표식 span은 래퍼째 숨기고(자리 차지 방지), 그 다음 형제만 밀어낸다. */
[data-testid="element-container"]:has(.f5-regen-anchor){{display:none}}
[data-testid="element-container"]:has(.f5-regen-anchor)
  +[data-testid="element-container"]{{margin-top:22px}}
.f5-detail-lead{{font-size:12.5px;color:#4A5C58;line-height:1.75;margin:0 0 14px}}
/* 평가항목 세부항목 소제목 — 이름과 점수를 한 줄에. */
.f5-dt-sub{{font-size:12.5px;font-weight:700;color:var(--text);margin:16px 0 6px;
  padding-top:12px;border-top:1px solid #E3ECE9}}
.f5-dt-sub span{{font-size:11px;font-weight:500;color:var(--text-3);margin-left:6px}}
/* '원인 : / 사업상 영향 : / 선정 이유 :' 처럼 라벨을 앞세운 블록.
   문단만 이어 붙이면 어디부터 다른 항목인지 구분이 안 된다. */
.f5-dt-row{{margin:0 0 12px}}
.f5-dt-row:last-child{{margin-bottom:0}}
.f5-dt-label{{font-size:12px;font-weight:700;color:var(--text);margin:0 0 4px}}
.f5-dt-body{{font-size:12.5px;color:#4A5C58;line-height:1.75;margin:0}}
/* 실행 항목·완료 기준은 여러 줄짜리 목록 — 가운뎃점으로 항목을 구분한다. */
.f5-dt-li{{position:relative;font-size:12.5px;color:#4A5C58;line-height:1.75;
  margin:0 0 5px;padding-left:11px}}
.f5-dt-li::before{{content:"·";position:absolute;left:2px;color:var(--text-3);font-weight:700}}
.f5-card-anchor,.f5-tabs-anchor,.f5-report-shell-anchor{{display:none}}
/* 마커 span은 display:none이어도 그걸 감싼 element-container는 세로 블록의
   gap(1rem)을 그대로 차지한다 — 컨테이너 안에서 제목이 16px 아래로 밀려 보였다.
   래퍼째 숨겨야 유령 간격이 사라진다(form-card-marker·newcand-marker와 같은 패턴). */
[data-testid="element-container"]:has(.f5-report-shell-anchor){{display:none}}
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"] > [data-testid="element-container"] .f5-report-shell-anchor){{
  background:#F7FBFA;border:1px solid #D5E3DF!important;border-radius:16px!important;
  box-shadow:none!important;margin-top:18px}}
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"] > [data-testid="element-container"] .f5-card-anchor){{
  background:#FFFFFF;border:1px solid #D6E3DF!important;border-left-width:4px!important;
  border-radius:12px!important;box-shadow:none!important}}
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"] > [data-testid="element-container"] .f5-accent-high){{border-left-color:#E56B5D!important}}
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"] > [data-testid="element-container"] .f5-accent-medium){{border-left-color:#D89428!important}}
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"] > [data-testid="element-container"] .f5-accent-low){{border-left-color:#5A86B8!important}}
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"] > [data-testid="element-container"] .f5-accent-maintain){{border-left-color:#2F8A63!important}}
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"] > [data-testid="element-container"] .f5-accent-unscored){{border-left-color:#9AAEAA!important}}
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"] > [data-testid="element-container"] .f5-accent-teal){{border-left-color:#2C7A73!important}}
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"] > [data-testid="element-container"] .f5-accent-short){{border-left-color:#D96B5F!important}}
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"] > [data-testid="element-container"] .f5-accent-mid){{border-left-color:#D89428!important}}
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"] > [data-testid="element-container"] .f5-accent-long){{border-left-color:#2C7A73!important}}
/* 제목과 배지를 한 줄에 — 배지가 제목 오른쪽에 붙어 카드가 짧아진다. */
.f5-card-head{{display:flex;flex-wrap:wrap;align-items:center;gap:6px;margin:0 0 2px}}
.f5-card-head .f5-card-title{{margin:0}}
/* 카드 안 펼치기 = 상자가 아니라 '+ 자세히 보기' 텍스트 링크로 보이게 한다.
   Streamlit 기본 expander는 테두리 있는 회색 상자라 카드 안에서 덩어리로 읽혔다. */
/* 카드 안은 기본 간격(1rem)이 넓어 본문과 펼치기 사이가 떴다. 카드 아래 여백도
   같이 줄여 '+ 자세히 보기' 밑이 허전해 보이지 않게 한다. */
{F5_CARD} > div > [data-testid="stVerticalBlock"]{{gap:0.5rem}}
{F5_CARD}{{padding-bottom:10px!important}}
{F5_CARD} [data-testid="stExpander"]{{background:transparent!important;border:none!important;
  border-radius:0!important;margin-top:4px}}
{F5_CARD} [data-testid="stExpander"] details{{background:transparent!important;border:none!important;
  border-top:1px solid #E3ECE9!important;border-radius:0!important}}
{F5_CARD} [data-testid="stExpander"] summary{{padding:9px 0 0!important;background:transparent!important}}
{F5_CARD} [data-testid="stExpander"] summary:hover{{background:transparent!important}}
{F5_CARD} [data-testid="stExpander"] summary p{{color:#2C7A73!important;font-size:12px!important;
  font-weight:700!important;margin:0!important}}
{F5_CARD} [data-testid="stExpander"] summary p::before{{content:"+ "}}
{F5_CARD} [data-testid="stExpander"] details[open] summary p::before{{content:"− "}}
/* 기본 꺾쇠 아이콘은 '+' 표기와 중복이라 숨긴다. */
{F5_CARD} [data-testid="stExpander"] summary svg{{display:none!important}}
[data-testid="stVerticalBlock"]:has(> [data-testid="element-container"] .f5-tabs-anchor)
  button[data-baseweb="tab"][aria-selected="true"]{{color:#2C7A73!important}}
[data-testid="stVerticalBlock"]:has(> [data-testid="element-container"] .f5-tabs-anchor)
  [data-baseweb="tab-highlight"]{{background-color:#2C7A73!important}}
.f5-timeline{{position:relative;display:grid;grid-template-columns:repeat(3,1fr);margin:12px 0 16px}}
.f5-timeline::before{{content:"";position:absolute;left:16.67%;right:16.67%;top:8px;
  height:1px;background:#C9D9D5}}
.f5-timeline-step{{position:relative;text-align:center;color:#536B68}}
.f5-timeline-dot{{position:relative;z-index:1;display:block;width:9px;height:9px;margin:4px auto 9px;
  background:#FFFFFF;border:2px solid #2C7A73;border-radius:50%}}
.f5-timeline-short .f5-timeline-dot{{border-color:#D96B5F}}
.f5-timeline-mid .f5-timeline-dot{{border-color:#D89428}}
.f5-timeline-long .f5-timeline-dot{{border-color:#2C7A73}}
.f5-timeline-label{{display:block;font-size:12px;font-weight:700;color:#173B3A;margin-bottom:2px}}
.f5-timeline-copy{{display:block;font-size:10.5px;color:#6C817D}}
.f5-phase{{font-size:11px;font-weight:700;letter-spacing:.03em;margin:0 0 8px}}
.f5-phase-short{{color:#C65A50}}
.f5-phase-mid{{color:#B47719}}
.f5-phase-long{{color:#2C7A73}}
/* 점수 산출 근거 — 항목별 서술형 판단 근거 (숫자 표 밑에 붙는 목록)
   Streamlit의 동적 emotion-cache 클래스가 'li{{font-size:1rem}}'을 넣는데
   '.클래스 li'와 특정도가 같아서 나중에 로드되는 쪽이 이긴다(순서 의존이라 불안정).
   'ul.reason-list li'로 특정도를 하나 더 올려 title(14px)과 항상 같게 만든다. */
.reason-list{{list-style:none;margin:0;padding:0;font-size:14px;color:var(--text-2)}}
ul.reason-list li{{font-size:14px;padding:8px 0 8px 14px;border-bottom:1px dashed var(--border);
  position:relative;line-height:1.55}}
.reason-list li:last-child{{border-bottom:none;padding-bottom:0}}
.reason-list li:first-child{{padding-top:0}}
.reason-list li::before{{content:"·";position:absolute;left:0;color:var(--text-3)}}
.reason-list b{{color:var(--text);font-weight:600;font-size:14px}}
.ob-chip.warn{{border-color:var(--warn)}}
.ob-chip.warn .ob-check{{color:var(--warn)}}
.wait-tag{{display:inline-block;font-size:10px;color:var(--warn);background:var(--warn-bg);
  padding:1px 6px;border-radius:5px;margin-left:6px}}
.stTextInput input, .stSelectbox div[data-baseweb="select"] > div{{
  border-radius:8px;border-color:var(--border);background:var(--surface)}}
</style>
<a class="ob-logo" href="?home=1" target="_self" title="처음 화면으로"></a>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# 화면 0 — 인트로
# ════════════════════════════════════════════════════════════
FEATURES = [
    ("""<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="4.5"/>
        <circle cx="12" cy="12" r="1.1" fill="#159A93" stroke="none"/>""",
     "신사업 추천", "조직 역량에 맞는 신사업 후보를 순위로 찾아줘요"),
    ("""<path d="M5 4h11l3 3v13a1 1 0 01-1 1H5a1 1 0 01-1-1V5a1 1 0 011-1Z"/>
        <path d="M8.5 13.5l2 2 4-4.5"/>""",
     "아이디어 적합도 판별", "생각해둔 아이디어가 우리 조직에 맞는지 채점해요"),
    ("""<path d="M4 19h4v-4H4z"/><path d="M10 19h4v-7h-4z"/><path d="M16 19h4V8h-4z"/>""",
     "부족 역량 보완 로드맵", "목표 점수까지 무엇을 채워야 할지 알려줘요"),
]


def screen_intro():
    feats = "".join(f"""
      <div class="feat">
        <span class="ico"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#159A93"
          stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round">{path}</svg></span>
        <span>
          <p class="t1">{title}</p>
          <p class="t2">{desc}</p>
        </span>
      </div>""" for path, title, desc in FEATURES)

    st.markdown(f"""
<p class="intro-h1">할 수 있는 신사업부터<br>부족한 역량까지 진단해요</p>
<p class="intro-sub">조직 자료만 올리면 AI가 <b>{len(ITEMS)}개 항목 100점 만점</b>으로 채점해요</p>
<div class="intro-card">{feats}</div>
""", unsafe_allow_html=True)

    if st.button("시작하기", key="start", type="primary"):
        go(1)


# ════════════════════════════════════════════════════════════
# 화면 1~3 — 파일 업로드
# ════════════════════════════════════════════════════════════
UPLOAD_STEPS = {
    1: {"key": "org_intro", "title": "조직소개서 파일 업로드", "types": ["pptx", "pdf"],
        "tips": ["PPTX 파일만 지원합니다.",
                 "최신 조직소개서 자료를 업로드하세요.",
                 "상용화 실적·협력 기관·전문 인력이 드러나면 더 정확해집니다.",
                 "최대 200MB까지 업로드할 수 있습니다."]},
    2: {"key": "hr_info", "title": "인력정보 파일 업로드", "types": ["xlsx", "pdf"],
        "tips": ["XLSX 파일만 지원합니다.",
                 "최신 인력 현황 자료를 업로드하세요.",
                 "담당 영역·경력이 구체적일수록 더 정확해집니다.",
                 "최대 200MB까지 업로드할 수 있습니다."]},
    3: {"key": "patent", "title": "특허목록 파일 업로드", "types": ["xlsx", "pdf"],
        "tips": ["XLSX 파일만 지원합니다.",
                 "최신 특허 목록을 업로드하세요.",
                 "특허 분류가 함께 있으면 더 정확해집니다.",
                 "최대 200MB까지 업로드할 수 있습니다."]},
}


def progress_bar(step: int) -> str:
    return ('<div class="ob-progress">'
            + "".join(f'<span class="{"on" if i <= step else ""}"></span>' for i in range(1, 5))
            + "</div>")


def screen_upload(step: int):
    """1-1 ~ 1-3 파일 업로드. 파일 내용을 bytes로 보관해야 F1이 파싱할 수 있다.

    ⚠️ 파일명만 저장하면 안 된다 (원래 코드의 문제). Streamlit의 UploadedFile은
       rerun 때마다 새로 만들어지고, 한 번 읽으면 위치가 끝으로 가서 재사용이
       안 된다. 그래서 여기서 bytes로 뽑아 세션에 넣는다.
    """
    cfg = UPLOAD_STEPS[step]
    st.markdown(progress_bar(step) + f'<p class="ob-step">STEP {step}</p>', unsafe_allow_html=True)

    saved = st.session_state.uploaded.get(cfg["key"])
    if saved:
        # ⚠️ st.file_uploader를 다시 그리지 않는다.
        #    Streamlit은 위젯이 그려지지 않은 rerun에서 그 위젯의 상태를 버린다.
        #    STEP 3 → 4 → 이전으로 돌아오면 업로더가 비어 있게 되므로,
        #    우리가 세션에 보관한 bytes로 같은 모양의 박스를 직접 그린다.
        #    (새로고침하면 세션 자체가 새로 생겨 여기도 비게 된다 — 의도한 동작)
        fname = st.session_state.files.get(cfg["key"], "")
        # ✕는 <a>로 만든다. st.button은 박스 밖에만 놓을 수 있어서
        # 원래 Streamlit 업로더처럼 칩 안에 넣을 수 없다.
        # 이 앱은 이미 후보 카드에서 같은 쿼리 파라미터 방식을 쓴다.
        st.markdown(
            f'<div class="up-box"><div class="up-file">'
            f'<span class="up-doc"></span>'
            f'<span class="up-name">{fname}</span>'
            f'<span class="up-size">{len(saved) / 1024:,.1f}KB</span>'
            f'<a class="up-x" href="?screen={step}&clear={cfg["key"]}" '
            f'target="_self" title="파일 삭제">&times;</a>'
            f"</div></div>", unsafe_allow_html=True)
    else:
        # 아이콘·제목은 CSS로 점선 박스 안에 그려진다 (inject_css의 upload_title)
        up = st.file_uploader(cfg["title"], type=cfg["types"], key=f"up_{cfg['key']}",
                              label_visibility="collapsed")
        if up is not None:
            st.session_state.files[cfg["key"]] = up.name
            st.session_state.uploaded[cfg["key"]] = up.getvalue()   # ← 내용 보관
            st.session_state.pop("profile", None)   # 파일이 바뀌면 프로필 다시 생성
            st.rerun()

    tips = "".join(f"<li>{t}</li>" for t in cfg["tips"])
    st.markdown(f'<div class="ob-tip"><p class="ob-tip-title"><i>&#9432;</i> 업로드 팁</p>'
                f"<ul>{tips}</ul></div>", unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        if st.button("← 이전", key=f"prev_{step}", type="secondary"):
            go(step - 1)
    with right:
        if st.button("다음으로 →", key=f"next_{step}", type="primary"):
            go(step + 1)


# ════════════════════════════════════════════════════════════
# 화면 4 — 분석 완료
# ════════════════════════════════════════════════════════════
#   (세션 키, 미업로드 시 표시 이름, 파싱 성공 시 표시 문구)
RESULT_ROWS = [
    ("org_intro", "조직소개서", "인력정보, 팀역량종합, 조직개요 추출 완료"),
    ("hr_info", "인력정보", "개인별 역량 추출 완료"),
    ("patent", "특허목록", "특허 목록 추출 완료"),
]


# 1-5  파싱 단계 → 어느 업로드 파일에서 나온 것인지
PARSE_SOURCE = {
    "조직소개서": "org_intro",
    "인력정보": "hr_info", "팀역량종합": "hr_info",
    "개인별역량": "hr_info", "조직개요": "hr_info",
    "특허목록": "patent",
}


def screen_done():
    """1-4 특허 보유 자동 판별 · 1-5 업로드/파싱 상태 시각화.

    여기서 F1을 실제로 실행한다. 업로드가 없으면 samples/로 대체된다.
    """
    profile = load_org_profile()
    status = (profile or {}).get("parse_status", {})

    chips = ""
    for key, label, done_note in RESULT_ROWS:
        name = st.session_state.files.get(key)
        # 이 파일에서 나온 파싱 단계들의 결과를 모은다
        steps = {k: v for k, v in status.items() if PARSE_SOURCE.get(k) == key}
        ok = [k for k, v in steps.items() if v["ok"]]
        bad = [(k, v["detail"]) for k, v in steps.items() if not v["ok"]]

        if not name and not ok:
            chips += (f'<div class="ob-chip off"><span class="ob-check">·</span>'
                      f'<div><p class="cname">{label} 미업로드</p>'
                      f'<p class="csub">건너뛴 항목이에요</p></div></div>')
            continue

        title = name or f"{label} (샘플 데이터)"
        if bad and not ok:
            mark, cls = "!", "ob-chip warn"
            sub = f'<b>파싱 실패</b> · {bad[0][1]}'
        elif bad:
            mark, cls = "&check;", "ob-chip"
            sub = f'{len(ok)}개 항목 추출 · {len(bad)}개 실패'
        else:
            mark, cls = "&check;", "ob-chip"
            sub = done_note
        chips += (f'<div class="{cls}"><span class="ob-check">{mark}</span>'
                  f'<div><p class="cname">{title}</p><p class="csub">{sub}</p></div></div>')

    st.markdown(progress_bar(4) + '<p class="ob-done-title">업로드 데이터 분석완료.</p>',
                unsafe_allow_html=True)
    # 점선 상자와 아래 문구를 한 번에 출력한다.
    # st.markdown을 두 번 부르면 Streamlit이 블록 사이에 48px 여백을 넣어 떨어져 보인다.
    if profile:
        caps = [c for c in profile["capabilities"] if c["evidence_ids"]]
        n_patent = sum(1 for e in profile["evidence"].values() if e["type"] == "patent")
        n_staff = len(profile["evidence_ids"]) - n_patent
        # 1-4  특허 보유 여부 자동 판별 결과를 문장으로
        patent_note = ("<b>확인됨</b> — 특허 기반 역량을 점수에 반영합니다."
                       if profile.get("has_patent_data")
                       else "<b>없음</b> — 인력정보 기반 역량만 반영합니다.")
        note = (f'보유 역량 <b>{len(caps)}개</b> · 특허 근거 <b>{n_patent}건</b> · '
                f'인력 근거 <b>{n_staff}건</b> 추출<br>특허 데이터 {patent_note}')
    elif not st.session_state.get("uploaded"):
        note = ("<b>업로드된 파일이 없습니다.</b><br>"
                "STEP 1~3에서 파일을 올려 주세요.")
    else:
        err = st.session_state.get("_f1_error", "")
        note = ("조직 데이터를 읽지 못했습니다. 시장계열 점수만 표시됩니다."
                + (f" ({err[:80]})" if err else ""))

    st.markdown(f'<div class="ob-result-box">{chips}</div>'
                f'<p class="ob-done-note">{note}</p>', unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        if st.button("← 이전", key="prev_4", type="secondary"):
            go(3)
    with right:
        if st.button("완료 →", key="to_dash", type="primary"):
            go(5)


# ════════════════════════════════════════════════════════════
# 화면 5 — 대시보드
# ════════════════════════════════════════════════════════════
def radar_svg(scores: list, labels: list, maxes: list) -> str:
    """시장계열 3개 항목 레이더 (onboarding_final.html의 radarSVG와 동일한 방식)"""
    n = len(labels)
    if n == 0:  # 에러 방어 추가
        return ""
    
    cx = cy = 110
    R = 70
    n = len(labels)

    def pt(i, r):
        ang = -math.pi / 2 + i * 2 * math.pi / n
        return cx + r * math.cos(ang), cy + r * math.sin(ang)

    rings = ""
    for f in (0.25, 0.5, 0.75, 1):
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in (pt(i, R * f) for i in range(n)))
        rings += f'<polygon points="{pts}" fill="none" stroke="#D5EDEB" stroke-width="0.5"/>'
    axes = "".join(
        f'<line x1="{cx}" y1="{cy}" x2="{x:.1f}" y2="{y:.1f}" stroke="#D5EDEB" stroke-width="0.5"/>'
        for x, y in (pt(i, R) for i in range(n)))
    data = " ".join(f"{x:.1f},{y:.1f}" for x, y in
                    (pt(i, R * (scores[i] / maxes[i] if maxes[i] else 0)) for i in range(n)))
    texts = "".join(
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="10" fill="#6C8A85" text-anchor="middle" '
        f'dominant-baseline="middle">{labels[i]}</text>'
        for i, (x, y) in enumerate(pt(i, R + 20) for i in range(n)))
    # preserveAspectRatio로 카드 높이에 맞춰 비율을 지키며 확대된다
    return (f'<svg viewBox="0 0 220 220" preserveAspectRatio="xMidYMid meet">{rings}{axes}'
            f'<polygon points="{data}" fill="rgba(21,154,147,0.16)" stroke="#0E837D" stroke-width="1.5"/>'
            f"{texts}</svg>")


FLAG_LABELS = {
    "has_domain_expert": "도메인 전문 인력",
    "inference_speed_ok": "추론 속도 요건",
    "has_partner_network": "파트너 네트워크",
    "has_commercialization_experience": "상용화 경험",
}


def render_org_summary(profile: dict) -> None:
    """F1 조직 역량 프로필. profile이 없으면 '연결 대기'로 남긴다."""
    if not profile:
        err = st.session_state.get("_f1_error", "")
        st.markdown(f"""
<div class="org-summary">
  <div>
    <div class="oname">조직 역량 프로필</div>
    <div class="osub">조직소개서 · 인력정보 · 특허목록에서 추출{f' — {err[:60]}' if err else ''}</div>
  </div>
  <div class="wait-tag" style="font-size:11px;padding:4px 10px">데이터 연결 대기</div>
</div>""", unsafe_allow_html=True)
        return

    caps = [c for c in profile["capabilities"] if c["evidence_ids"]]
    ctx = profile.get("org_context", {})
    n_patent = sum(1 for e in profile.get("evidence", {}).values()
                   if e.get("type") == "patent")

    chips = "".join(
        f'<span class="cap-chip"><b>{c["name"]}</b>'
        f'<i>{"★" * (c["level"] or 0)}</i>'
        f'<u>근거 {len(c["evidence_ids"])}</u></span>'
        for c in sorted(caps, key=lambda x: -(x["level"] or 0)))

    flags = "".join(
        f'<span class="ctx-flag {"on" if ctx.get(k) else "off"}">'
        f'{"✓" if ctx.get(k) else "—"} {label}</span>'
        for k, label in FLAG_LABELS.items())

    # 연결된 상태는 배지를 붙이지 않는다. 정상 동작은 조용한 게 맞고,
    # 배지는 '연결 대기'처럼 문제가 있을 때만 눈에 띄어야 한다.
    st.markdown(f"""
<div class="org-summary" style="display:block">
  <div>
    <div class="oname">조직 역량 프로필</div>
    <div class="osub">보유 역량 {len(caps)}개 · 특허 근거 {n_patent}건 · 인력 근거
      {len(profile["evidence_ids"]) - n_patent}건</div>
  </div>
  <div class="cap-row">{chips}</div>
  <div class="ctx-row">{flags}</div>
</div>""", unsafe_allow_html=True)

    # ── 2-3  저장된 프로필 확인 ──
    render_profile_inspector(profile)


def render_profile_inspector(profile: dict) -> None:
    """F2. 조직 역량 프로필 — 2-1 추출 정보 확인 · 2-2 표준 스키마 저장."""
    meta = profile.get("_meta", {})
    counts = profile_schema.summary_counts(profile)
    ok, errors, warnings = profile_schema.validate_profile(profile)

    label = "조직 역량 프로필 확인"
    if errors:
        label += f" · 오류 {len(errors)}건"
    elif warnings:
        label += f" · 경고 {len(warnings)}건"

    with st.expander(label):
        cols = st.columns(len(counts))
        for col, (k, v) in zip(cols, counts.items()):
            col.metric(k, v)

        st.caption(f"생성 {meta.get('saved_at', '—')}")
        if meta.get("source") == "samples":
            st.warning("개발 모드(`?dev=1`) — samples/ 폴더의 샘플 데이터로 만든 "
                       "프로필입니다. 실제 업로드 결과가 아닙니다.")

        # ── 2-2  표준 스키마 검증 ──
        #    통과했을 때는 아무것도 띄우지 않는다(정상은 조용해야 한다).
        #    깨졌을 때만 알린다 — 그때는 점수 계산이 틀어지므로 반드시 보여야 한다.
        if errors:
            st.error("스키마 오류 — 이 상태로는 점수 계산이 깨집니다\n\n"
                     + "\n".join(f"- {e}" for e in errors))
        elif warnings:
            st.warning("\n".join(f"- {w}" for w in warnings))

        # ── 2-1  추출 정보 확인 (4개 카테고리) ──
        tabs = st.tabs([t for t, _, _ in profile_schema.PREVIEW_SECTIONS])
        for tab, (title, fn, source) in zip(tabs, profile_schema.PREVIEW_SECTIONS):
            with tab:
                df = fn(profile)
                st.caption(f"출처: {source}")
                if df.empty:
                    st.info("추출된 항목이 없습니다.")
                else:
                    st.dataframe(df, hide_index=True, use_container_width=True)



def _usd_label(v) -> str:
    """억달러 값을 짧게 표기. None이면 '미확인'."""
    if v is None or pd.isna(v):
        return "미확인"
    return f"{v:,.0f}억$" if v >= 100 else f"{v:,.1f}억$"


def candidate_card(row, rank: int, picked: bool) -> str:
    """순위 카드.

    ⚠️ <a href="?pick=N">를 쓰면 안 된다. 링크는 페이지를 새로 로드하고,
       그러면 Streamlit 세션이 새로 만들어져 st.session_state가 통째로 비워진다
       (업로드한 조직 프로필까지 사라진다 — 실제로 2위 카드를 누르면
       '조직 역량 프로필'이 F1 연결 대기로 돌아갔다).
       대신 카드 위에 투명한 st.button을 겹쳐 두고 rerun으로 처리한다.
    """
    total = row["총점"]
    usd = _usd_label(row["시장규모_억달러"])
    den = row["배점합"]
    scope = "8개 항목 100점 환산" if den >= 100 else f"미평가 항목 제외 · 분모 {den}점"
    return f"""
<div class="card {'pick' if picked else ''}">
  <span class="rank">{rank}위 · DB 매칭</span>
  <p class="cname">{row['아이디어명']}</p>
  <div class="cscore-row">
    <div class="gauge" style="background:conic-gradient(var(--accent) {total:.0f}%, var(--surface-2) 0)">
      <span>{total:.0f}</span></div>
    <div>
      <div class="cscore">{total:.1f}<span class="cmax">/100</span></div>
      <div class="csub cmeta">진입장벽 {row['진입장벽_등급']} · 시장 {usd}</div>
    </div>
  </div>
  <span class="tag">{scope}</span>
</div>"""


def scorecard_table(row) -> str:
    """8개 항목 스코어카드.

    값이 None인 항목은 '평가 대상 없음'이다(예: 입력유형 역량을 전혀 요구하지 않는
    사업). 0점으로 표시하면 '나쁨'으로 읽히므로 —로 두고 분모에서 뺀다.
    """
    body = ""
    for name, cap, part in ITEMS:
        val = row[name]
        if val is None or (isinstance(val, float) and pd.isna(val)):
            note = "" if (f3_4 or part == "F2") else '<span class="wait-tag">평가 모듈 미연결</span>'
            body += (f'<tr class="wait"><td>{name}{note}</td>'
                     f"<td>{cap}</td><td>—</td></tr>")
        else:
            body += (f'<tr><td>{name}<div class="bar-track"><div class="bar-fill" '
                     f'style="width:{val / cap * 100:.0f}%"></div></div></td>'
                     f'<td>{cap}</td><td class="score-cell">{val}</td></tr>')
    return (f'<table class="sc-table"><tr><th>평가 항목</th><th>배점</th><th>획득</th></tr>'
            f"<tbody>{body}</tbody></table>")


def build_score_reasons(sel) -> list:
    """항목별 판단 근거 문장.

    LLM을 쓰지 않는다 — 표준역량_정의.csv의 '근거'·'전이출처'·'전담인력' 컬럼과
    DB 원문(시장규모·진입장벽수준·실제기업사례)에서 그대로 뽑아 조립한다.
    숫자만 있던 '점수 산출 근거' 카드에 그 점수가 나온 이유를 붙이기 위한 것.

    Returns: [(항목명, 근거 문장, 출처 태그), ...] — 평가되지 않은(None) 항목은 뺀다.
    """
    if not f3_2:
        return []

    matched_ids = list(sel["matched"] or [])
    missing_ids = list(sel["missing"] or [])

    def texts_for(ids, field, limit=2):
        seen, out = set(), []
        for cid in ids:
            row = f3_2.cap_row(cid)
            v = row.get(field) if row is not None else None
            if row is None or pd.isna(v):
                continue
            v = str(v).strip()
            if not v or v in seen:
                continue
            seen.add(v)
            out.append(v)
            if len(out) >= limit:
                break
        return out

    # 부족역량 중 '전이출처'가 보유역량을 가리키면 도메인만 바꾸면 재사용 가능하다
    transfer_lines, transfer_ids = [], set()
    for cid in missing_ids:
        row = f3_2.cap_row(cid)
        src = row.get("전이출처") if row is not None else None
        if row is None or pd.isna(src):
            continue
        src_ids = [s.strip() for s in str(src).split("+")]
        if all(f3_2.is_held(s) for s in src_ids):
            transfer_ids.add(cid)
            transfer_lines.append(
                f"'{f3_2.cap_name(cid)}'은(는) 보유 중인 "
                f"'{'·'.join(f3_2.cap_name(s) for s in src_ids)}' 역량에서 "
                "도메인만 전환하면 재사용 가능합니다.")

    gap_ids = [c for c in missing_ids if c not in transfer_ids]
    staffed = texts_for(matched_ids, "전담인력", limit=3)

    def market_text(*fields):
        parts = [str(sel[f]) for f in fields if pd.notna(sel.get(f))]
        return " — ".join(parts)

    by_item = {
        "조직역량적합도": (" · ".join(texts_for(matched_ids, "근거", 2))
                     or "요구역량과 직접 일치하는 보유 역량이 없습니다.", "조직 프로필"),
        "역량전이가능성": (" ".join(transfer_lines[:2])
                     or "보유 역량에서 바로 전환 가능한 요구역량이 확인되지 않았습니다.", "조직 프로필"),
        "부족역량수준": (
            f"{', '.join(f3_2.cap_names(gap_ids)[:4])} 등 {len(gap_ids)}개 역량이 "
            "보유 이력 없이 부족합니다." if gap_ids else "부족한 역량이 확인되지 않았습니다.",
            "조직 프로필"),
        "실행가능성": (
            f"{', '.join(dict.fromkeys(staffed))} 등 보유 인력으로 즉시 투입 가능합니다."
            if staffed else "요구역량에 배정된 전담인력이 없어 추가 확보가 필요합니다.",
            "조직 프로필"),
        "시장성": (market_text("시장규모"), "신사업DB"),
        "경쟁강도": (market_text("실제기업사례", "기업사례_요약"), "신사업DB"),
        "진입장벽": (market_text("진입장벽수준"), "신사업DB"),
        "사업성": (market_text("실제기업사례", "기업사례_요약"), "신사업DB"),
    }

    out = []
    for name, _, _ in ITEMS:
        if pd.isna(sel.get(name)):
            continue
        text, tag = by_item.get(name, ("", ""))
        if text:
            out.append((name, text, tag))
    return out


@st.dialog(" ")
def _db_source_dialog(sel) -> None:
    """'DB 원문보기' 팝업. st.dialog가 우상단 X를 기본 제공하므로 따로 안 만든다.

    제목을 빈 문자열로 주면 안 열려서(ValueError) 공백 하나(" ")를 넣어
    타이틀 바를 비워 보이게 한다.
    """
    db_rows = [
        ("설명", sel["설명"]),
        ("산업분류", sel["산업분류"]),
        ("필요역량태그", sel["필요역량태그"]),
        ("실제기업사례", f"{sel['실제기업사례']} — {sel['기업사례_요약']}"),
        ("시장규모", sel["시장규모"]),
    ]
    st.markdown(f"""
<div class="db-grid">
  {"".join(f'<span class="db-k">{k}</span><span class="db-v">{v}</span>' for k, v in db_rows)}
</div>""", unsafe_allow_html=True)


def _find_empty_idea_tab() -> int:
    """4필드가 전부 빈 탭 번호를 앞에서부터 찾는다. 없으면 새 탭을 만들어서 반환한다.

    6C(F4-3) → 5A 연결 시 기존 작성 내용을 지우지 않으려고 쓴다 — 1탭부터
    확인해서 비어 있는 첫 탭에 넣고, 전부 채워져 있으면 탭을 하나 늘린다.
    """
    count = st.session_state.get("idea_tab_count", 3)
    tab_data_all = st.session_state.get("idea_tab_data", {})
    for i in range(1, count + 1):
        fields = tab_data_all.get(i, {})
        if not any(str(v).strip() for v in fields.values()):
            return i
    new_count = count + 1
    st.session_state["idea_tab_count"] = new_count
    return new_count


def render_idea_tab(db: pd.DataFrame, profile: dict | None, org_ctx: dict):
    """아이디어 적합도 판단형 (핵심기능 2) — 5A → 6A → 7 → 8 → 9~11.

    5A  F4-1  정형 입력 4필드 → 요구역량 명세 (로컬 임베딩 + 필요시 LLM 보완)
    6A  F2-2  가장 가까운 DB 항목을 찾아 시장 데이터(시장규모·진입장벽)를 빌려온다
    7   F3-4  요구역량 + 그 시장 데이터로 8개 항목 100점 채점
    6A' F4-2  LLM이 전이 가능성까지 보고 판단 근거를 서술
    9~11 F5   갭 리포트·보완전략·로드맵

    탭 여러 개를 둬서(idea_active_tab) 아이디어를 새로 쓰거나 6C에서 넘어와도
    이전 탭에서 작성·채점한 내용이 지워지지 않게 한다. 위젯 key와 결과 캐시
    key를 전부 탭 번호로 구분해서, 탭마다 독립된 세션처럼 동작한다.
    """
    st.session_state.setdefault("idea_tab_count", 3)
    st.session_state.setdefault("idea_active_tab", 1)
    tab_count = st.session_state["idea_tab_count"]
    active_tab = st.session_state["idea_active_tab"]

    # ── 탭 줄 — 카드 '위' 오른쪽에 얹는다 (브라우저 탭처럼) ──
    # 예전엔 카드 안에서 배지 옆에 뒀는데, 입력 폼 폭을 깎아먹고 배지와 높이도
    # 안 맞았다. 카드 바깥 위로 빼면 '이 카드가 여러 장 중 하나'라는 게 바로 읽힌다.
    # 뒤따르는 stHorizontalBlock을 CSS로 찾기 위한 표식(화면엔 안 보인다).
    st.markdown('<span class="idea-tabs-marker" style="display:none"></span>',
                unsafe_allow_html=True)
    tab_btn_cols = st.columns(tab_count + 1)
    for i in range(1, tab_count + 1):
        with tab_btn_cols[i - 1]:
            if st.button(str(i), key=f"idea_tab_sel_{i}",
                         type="primary" if i == active_tab else "secondary"):
                st.session_state["idea_active_tab"] = i
                st.rerun()
    with tab_btn_cols[tab_count]:
        # 반각 "+"는 버튼 라벨(마크다운)에서 리스트 기호로 해석돼 빈 텍스트가 된다.
        # 전각 "＋"(U+FF0B)는 마크다운 문법이 아니라 글자 그대로 찍힌다.
        if st.button("＋", key="idea_tab_plus"):
            st.session_state["idea_tab_count"] = tab_count + 1
            st.session_state["idea_active_tab"] = tab_count + 1
            st.rerun()

    # ── 5A 입력 폼 (F4-1 정형 입력 양식) ──
    with st.container(border=True):
        # 이 카드를 CSS에서 찾기 위한 표식(화면엔 안 보인다).
        st.markdown('<span class="form-card-marker" style="display:none"></span>',
                    unsafe_allow_html=True)
        st.markdown('<span class="form-badge">아이디어를 작성해주세요.</span>',
                    unsafe_allow_html=True)

        # Streamlit은 위젯이 이번 실행에서 렌더링되지 않으면(=다른 탭을 보고 있으면)
        # 그 key의 session_state를 지워버린다("위젯이 스크립트에서 빠지면 상태도
        # 사라진다" — 실측으로 확인함, 2026-08-06). key만 믿으면 탭을 옮겼다가
        # 돌아왔을 때 입력이 사라진다. 그래서 위젯과 별개로 idea_tab_data에
        # 직접 값을 들고 있다가, 위젯을 그릴 때마다 value=로 재주입한다.
        st.session_state.setdefault("idea_tab_data", {})
        tab_data = st.session_state["idea_tab_data"].setdefault(
            active_tab, {"industry": "", "market": "", "problem": "", "customer": ""})

        # value=로 예시를 박아 두면 사용자가 안 쓴 문장이 이미 채워져 있어서
        # 자기 아이디어를 채점한 결과인지 데모인지 구분할 수 없으므로,
        # placeholder는 그대로 두고 value만 탭별 저장값으로 재주입한다.
        c1, c2 = st.columns(2)
        with c1:
            industry = st.text_input(
                "진출 산업", key=f"idea_industry_{active_tab}", value=tab_data["industry"],
                placeholder="예) 수의·반려동물 헬스케어")
        with c2:
            market = st.text_input(
                "목표 시장", key=f"idea_market_{active_tab}", value=tab_data["market"],
                placeholder="예) 국내 동물병원 시장")
        problem = st.text_area(
            "해결 문제", key=f"idea_problem_{active_tab}", value=tab_data["problem"], height=68,
            placeholder="예) 동물병원의 X-ray 판독을 원격으로도 실시간 지원해, "
                        "야간·소규모 병원의 진단 공백을 줄이고 싶음")
        customer = st.text_input(
            "목표 고객", key=f"idea_customer_{active_tab}", value=tab_data["customer"],
            placeholder="예) 1~2인 소규모 동물병원, 야간진료 특화 동물병원")

        # 이번 실행에서 위젯이 돌려준 최신값을 영구 저장소에 다시 써 둔다.
        tab_data["industry"], tab_data["market"] = industry, market
        tab_data["problem"], tab_data["customer"] = problem, customer

        fields = [industry, market, problem, customer]
        n_filled = sum(1 for v in fields if v and v.strip())
        ready = n_filled == len(fields)

        st.markdown(
            f'<p class="form-hint">4개 항목을 채우고 아래 버튼을 누르면 '
            f'8개 항목으로 채점합니다.'
            f'{"" if ready else f" (현재 {n_filled}/{len(fields)})"}</p>',
            unsafe_allow_html=True)
        run = st.button("적합도 판단하기", key=f"run_idea_fit_{active_tab}", type="primary",
                        disabled=not ready)

    if not f4_1:
        st.markdown("""
<div class="note-card wait">
  <p class="section-title">요구역량 명세 — 준비중</p>
  <p>요구역량 추출 모듈을 불러오지 못했습니다.</p>
</div>""", unsafe_allow_html=True)
        return

    user_input = {"industry": industry, "problem": problem,
                  "market": market, "customer": customer}

    req_key = f"idea_req_{active_tab}"
    # 채점은 '적합도 판단하기'를 눌렀을 때만 돈다. 6C(F4-3)에서 넘어오면 요구역량이
    # 미리 채워진 채로 도착하는데, 예전엔 그것만으로 화면 진입 즉시 채점이 시작돼서
    # 버튼을 누르지도 않았는데 도는 것처럼 보였다(2026-08-11).
    run_key = f"idea_run_{active_tab}"

    if run:
        prev = st.session_state.get(req_key)
        # 6C에서 넘어온 요구역량(F4-3 판단)은 폼 내용이 그때와 같으면 그대로 쓴다.
        # F4-1을 다시 태우면 조직이 이미 가진 역량이 순환 재검출돼 역량전이가능성이
        # 0점으로 깔린다 — F4-3 결과를 재사용하는 이유와 같다.
        if not (prev and prev.get("input") == user_input):
            with st.spinner("요구역량 추출 중… (최초 1회는 임베딩 모델 로딩으로 오래 걸립니다)"):
                try:
                    st.session_state[req_key] = f4_1.extract_idea_requirements(user_input)
                    st.session_state.pop(f"idea_req_error_{active_tab}", None)
                except Exception as e:
                    st.session_state[req_key] = None
                    st.session_state[f"idea_req_error_{active_tab}"] = str(e)
        st.session_state.pop(f"idea_judgment_{active_tab}", None)
        # 새로 판단을 요청했으니 이전 채점(캐시)은 버려서 다시 계산하게 한다.
        st.session_state.pop(f"idea_score_{active_tab}", None)
        st.session_state[run_key] = True
        st.rerun()

    req = st.session_state.get(f"idea_req_{active_tab}")
    if not req:
        err = st.session_state.get(f"idea_req_error_{active_tab}")
        st.markdown(f"""
<div class="note-card wait">
  <p class="section-title">요구역량 명세</p>
  <p>{'추출 실패 — ' + html.escape(err[:200]) if err
      else '위 4개 항목을 작성하고 <b>적합도 판단하기</b>를 눌러 주세요.'}</p>
</div>""", unsafe_allow_html=True)
        return

    # ── 7A F4-4로 채점 (조직계열=F3-2 그대로, 시장계열=LLM 추정 + F2-5 공식) ──
    # F4-1이 F3-2와 같은 표준역량 축(CAP_*)을 직접 반환한다.
    req_names = req["required_capability_names"]
    std_ids = req["required_capability_ids"]
    unmatched_caps = req.get("unmatched_capabilities") or []
    idea_context = {"industry": industry, "problem": problem, "market": market, "customer": customer}

    # ⚠️ 캐시 필수 — 안 하면 다른 탭 갔다 오는 것처럼 이 화면이 한 번이라도
    # 다시 그려질 때마다(=Streamlit rerun마다) LLM 시장 추정을 또 부르게 된다.
    # 매번 값이 달라지니(LLM 비결정성) "결과가 사라진다/바뀐다"로 보이고, API도
    # 계속 소모된다(2026-08-06 확인). "적합도 판단하기"를 다시 누르기 전까진
    # 같은 탭에서는 한 번 계산한 채점을 그대로 재사용한다.
    score_key = f"idea_score_{active_tab}"
    if score_key not in st.session_state:
        if not st.session_state.get(run_key):
            # 요구역량만 채워진 상태(6C에서 넘어옴) — 버튼을 눌러야 채점을 시작한다.
            st.markdown("""
<div class="note-card wait">
  <p class="section-title">요구역량 명세</p>
  <p>추천받은 아이디어를 위 폼에 채웠습니다.
     <b>적합도 판단하기</b>를 누르면 8개 항목으로 채점합니다.</p>
</div>""", unsafe_allow_html=True)
            return
        if f4_4 and (std_ids or unmatched_caps):
            with st.spinner("시장 데이터 추정 및 채점 중…"):
                st.session_state[score_key] = f4_4.calculate_idea_score(
                    std_ids,
                    idea_context,
                    org_ctx,
                    return_detail=True,
                    unmatched_capabilities=unmatched_caps,
                )
        else:
            st.session_state[score_key] = None

    r = st.session_state[score_key]
    if r:
        scores, det = r["scores"], r["detail"]
        total, raw, den = r["total_score"], r["raw_score"], r["denominator"]
        market_est = r["market_estimate"] or {}
        transfer_bonus = r.get("transfer_bonus") or {"applied": False}
    else:
        scores = {n: None for n, _, _ in ITEMS}
        det = {"matched": [], "missing": []}
        total, raw, den = 0.0, 0.0, 0
        market_est = {}
        transfer_bonus = {"applied": False}

    # scorecard_table·render_gap_report가 쓰는 필드를 갖춘 행으로 만든다.
    # 시장규모·진입장벽은 DB 실측이 아니라 AI 추정치다 — 화면에서 그렇게 표시한다.
    sel = pd.Series({
        **{n: scores.get(n) for n, _, _ in ITEMS},
        "아이디어ID": f"idea_input_{active_tab}", "아이디어명": f"{industry} · {problem[:30]}",
        "설명": problem,
        "필요역량태그": ", ".join(f3_2.cap_names(std_ids) if (f3_2 and std_ids)
                                 else req_names),
        "시장규모_억달러": market_est.get("market_size_usd"),
        "진입장벽_등급": market_est.get("entry_barrier"),
        "총점": total, "획득": raw, "배점합": den,
        "matched": det["matched"], "missing": det["missing"],
        "sub_scores": det.get("sub_scores", {}),
        "출처링크": None,
        "아이디어입력": idea_context,
        "LLM시장추정": market_est,
        "목록밖역량": unmatched_caps,
        "역량전이보너스": transfer_bonus,
    })

    barrier_txt = market_est.get("entry_barrier") or "—"
    market_note = (
        f'<p class="csub" style="margin-top:8px;color:var(--text-3)">'
        f'AI 시장 추정 근거 · {_safe_llm_text(market_est["rationale"])}</p>'
        if market_est.get("rationale") else
        (f'<p class="csub" style="margin-top:8px;color:var(--warn)">'
         f'시장 추정 실패 — {html.escape(str(market_est["error"]))} '
         f'(조직계열 점수만 반영됨)</p>' if market_est.get("error") else "")
    )
    transfer_bonus_note = (
        f'<p class="csub" style="margin-top:8px;color:var(--warn)">'
        f'역량전이가능성 {transfer_bonus["amount"]:.1f}점은 표준 목록 밖 역량 '
        f'{transfer_bonus["item_count"]}개에 대한 임시 보너스입니다. 항목당 고정 '
        f'{f4_4.UNMATCHED_BONUS_PER_ITEM if f4_4 else 2}점이며 표준 점수와 구분해 해석해야 합니다.</p>'
        if transfer_bonus.get("applied") else ""
    )
    st.markdown(f"""
<div class="canvaswrap">
  <p class="section-title">입력한 아이디어 — 적합도 판단 결과</p>
  <div class="cscore-row" style="margin-bottom:14px">
    <div class="gauge" style="width:60px;height:60px;background:conic-gradient(var(--accent) {total:.0f}%, var(--surface-2) 0)">
      <span style="font-size:14px">{total:.0f}</span></div>
    <div>
      <div style="font-size:26px;font-weight:600">{total:.1f}<span
        style="font-size:13px;color:var(--text-3)">/100</span></div>
      <div class="csub">획득 {raw:.1f}/{den}점 · 진입장벽(AI 추정) {barrier_txt}</div>
    </div>
  </div>
  <p class="csub" style="margin-bottom:12px">
    조직계열: 입력 아이디어의 요구역량 {len(std_ids)}개 기준<br>
    시장계열: DB 실측이 아니라 <b>AI가 추정</b>한 시장 데이터
  </p>
  {scorecard_table(sel)}
  {transfer_bonus_note}
  {market_note}
</div>""", unsafe_allow_html=True)
    if r and st.button("시장 데이터 다시 추정하기", key=f"rescore_{active_tab}"):
        st.session_state.pop(score_key, None)
        st.rerun()

    # ── 요구역량 명세 + 판단 근거 ──
    left, right = st.columns([1, 1])
    with left:
        chips = "".join(
            f'<span class="cap-chip"><b>{html.escape(c["name"])}</b>'
            f'<u>{c["source"]}</u></span>' for c in req["required_capabilities"])
        llm_note = (f'<p class="csub" style="margin-top:8px;color:var(--text-3)">'
                    f'LLM 보완 사용 · {_safe_llm_text(req["llm_rationale"] or "")}</p>'
                    if req["used_llm_fallback"] else "")
        unmatched_note = (
            '<p class="csub" style="margin-top:8px;color:var(--warn)">'
            '표준 목록에 없는 역량(AI 판단) · '
            + html.escape(", ".join(unmatched_caps)) + "</p>"
        ) if unmatched_caps else ""
        # min-height는 레이더가 읽힐 만한 크기를 보장한다. 두 카드 모두 eq-card라
        # 내용이 더 많은 쪽 높이에 맞춰 함께 늘어난다(둘은 항상 같은 높이).
        st.markdown(f"""
<div class="canvaswrap eq-card" style="min-height:330px">
  <p class="section-title">요구역량 명세</p>
  <p class="csub">신뢰도 {req["confidence"]} · {len(req["required_capabilities"])}개 역량</p>
  <div class="cap-row">{chips}</div>
  {llm_note}
  {unmatched_note}
</div>""", unsafe_allow_html=True)
    with right:
        # 레이더는 .radar-wrap으로 감싸야 카드 높이를 끌어올리지 않고 카드에 맞춰 늘어난다
        # (역량 기반 추천형의 '점수 분포' 카드와 같은 구조)
        rad = [(n, m) for n, m, _ in ITEMS if sel[n] is not None and not pd.isna(sel[n])]
        st.markdown(f'<div class="canvaswrap eq-card" style="min-height:330px">'
                    f'<p class="section-title">{len(rad)}개 항목 점수 분포</p>'
                    f'<div class="radar-wrap">'
                    f'{radar_svg([sel[n] for n, _ in rad], [n for n, _ in rad], [m for _, m in rad])}'
                    "</div></div>", unsafe_allow_html=True)

    render_idea_judgment(sel, profile, tab=active_tab)
    with st.container(border=True):
        st.markdown('<span class="f5-report-shell-anchor"></span>', unsafe_allow_html=True)
        render_gap_report(
            sel,
            profile,
            org_ctx,
            key_prefix="idea_gap",
            diagnosis_mode="idea_fit",
        )

    if st.button("← 처음으로", key="back_ob_f2", type="secondary"):
        go(0)


def render_idea_judgment(sel, profile: dict | None, tab: int = 1) -> None:
    """6A 후반 — LLM 매칭 판단 (F4-2). 점수 뒤에 붙는 정성 근거."""
    if not f4_2 or not profile:
        return

    key = f"idea_judgment_{tab}"
    saved = st.session_state.get(key)

    if saved is None:
        if st.button("판단 근거 생성", key=f"gen_idea_judgment_{tab}",
                     use_container_width=True):
            cand = [{
                "id": str(sel["아이디어ID"]), "name": str(sel["아이디어명"]),
                "total_score": float(sel["총점"]),
                "scores": {n: sel[n] for n, _, _ in ITEMS},
                "matched": list(sel["matched"]), "missing": list(sel["missing"]),
            }]
            with st.spinner("LLM 매칭 판단 중…"):
                st.session_state[key] = f4_2.llm_match_judgment(profile, cand)
            st.rerun()
        return

    ranked = saved["ranked_candidates"]
    top = ranked[0] if ranked else {}
    # 파트 번호 배지(F4-2)는 뺀다. 다만 LLM이 실제로 돌지 않았을 때는
    # 아래 문장이 LLM 판단이 아니라는 뜻이므로 그 사실만 남긴다.
    fallback = "" if saved["used_llm"] else (
        '<p class="csub" style="margin-top:6px;color:var(--warn)">'
        'LLM 미사용 — 점수 순서를 그대로 유지했습니다.</p>')
    st.markdown(f"""
<div class="note-card">
  <p class="section-title">판단 근거</p>
  <p>{_safe_llm_text(strip_evidence_ids(top.get("recommendation_reason", "")))}</p>
  <p style="margin-top:8px"><b>기술 전이 가능성</b> ·
    {_safe_llm_text(strip_evidence_ids(top.get("transferability_note", "")) or "—")}</p>
  {fallback}
  {f'<p class="csub" style="margin-top:6px;color:var(--warn)">{html.escape(str(saved["error"]))}</p>' if saved.get("error") else ""}
</div>""", unsafe_allow_html=True)
    if st.button("다시 생성", key=f"regen_idea_judgment_{tab}"):
        st.session_state.pop(key, None)
        st.rerun()


def render_new_candidates(results: pd.DataFrame, profile: dict | None, all_candidates: pd.DataFrame) -> None:
    """6B 후반 — DB 밖 신규 후보 LLM 제안 (F4-3).

    DB 50건에 없는 아이디어라 점수 척도(시장규모·진입장벽)가 없다. 그래서 순위에
    섞지 않고 별도 카드로 보여준다. F4-3이 required_capability_names를 주므로
    나중에 F3-2로 조직계열 점수만 따로 낼 수는 있다.
    """
    if not f4_3 or not profile:
        return

    key = "new_cands"
    saved = st.session_state.get(key)

    with st.container(border=True):
        # 이 카드를 CSS에서 찾기 위한 표식 (화면에는 안 보인다)
        st.markdown('<span class="newcand-marker" style="display:none"></span>',
                    unsafe_allow_html=True)

        if saved is None:
            st.markdown("""
<p class="section-title">LLM추천 후보</p>
<p class="csub">신사업 DB 50건에 없는 아이디어를 조직 역량만 보고 LLM이 제안합니다.</p>
""", unsafe_allow_html=True)
            # 제안 중에는 버튼을 진행표시로 '바꿔 끼운다'. st.empty()에 다시 그리면
            # 버튼이 있던 자리가 그대로 교체돼서, 버튼과 '신규 후보 제안 중…'이
            # 위아래로 같이 보이지 않는다(중복으로 눌리는 것도 막힌다).
            # 갭 리포트 생성 버튼과 같은 방식이다.
            gen_slot = st.empty()
            if gen_slot.button("신규 후보 제안받기", key="gen_new_cands", type="primary"):
                # Top-3(results)가 아니라 DB 50건 전체(all_candidates)를 기준으로
                # 중복을 걸러야, 화면에 안 보이는 나머지 후보와 겹치는 제안을 막을 수 있다.
                existing = [{"id": r["아이디어ID"], "name": r["아이디어명"],
                             "total_score": r["총점"]} for _, r in all_candidates.iterrows()]
                with gen_slot.container():
                    with st.spinner("신규 후보 제안 중…"):
                        st.session_state[key] = f4_3.propose_new_candidates(profile, existing)
                st.rerun()
            return

        if saved.get("error"):
            st.markdown(f"""
<p class="section-title">LLM추천 후보 — 실패</p>
<p class="csub">{html.escape(str(saved["error"]))}</p>
""", unsafe_allow_html=True)
        elif not saved["new_candidates"]:
            st.markdown("""
<p class="section-title">LLM추천 후보</p>
<p class="csub">근거가 충분한 신규 아이디어를 찾지 못했습니다. (억지로 채우지 않습니다)</p>
""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
<p class="section-title">LLM추천 후보 {len(saved["new_candidates"])}건</p>
<p class="csub" style="margin-bottom:10px">{_safe_llm_text(strip_evidence_ids(saved.get("overall_summary", "")))}</p>
""", unsafe_allow_html=True)
            # 카드마다 버튼을 붙여야 해서(raw HTML엔 st.button을 못 심는다)
            # 카드 하나씩 개별 렌더링으로 바꿨다 — 이전엔 전부 합쳐서 한 번에 그렸었다.
            for c in saved["new_candidates"]:
                caps = ", ".join(c.get("required_capability_names", []))
                ev = c.get("cited_evidence_ids") or []
                st.markdown(
                    f'<div class="ob-chip" style="align-items:flex-start">'
                    f'<span class="ob-check">+</span><div>'
                    f'<p class="cname">{html.escape(c["name"])}</p>'
                    f'<p class="csub">{_safe_llm_text(strip_evidence_ids(c["description"]))}</p>'
                    f'<p class="csub" style="margin-top:6px">'
                    f'{_safe_llm_text(strip_evidence_ids(c["recommendation_reason"]))}</p>'
                    f'<p class="csub" style="color:var(--text-3);margin-top:6px">필요역량 {html.escape(caps)}'
                    f'{f" · 근거 {len(ev)}건" if ev else ""}</p>'
                    f"</div></div>", unsafe_allow_html=True)
                if st.button("이 아이디어로 적합도 판단하기", key=f"use_new_{c['id']}"):
                    # 6C(F4-3) → 5A(F4-1 입력 폼)로 이어주기.
                    #
                    # 예전엔 F4-3의 required_capability_ids를 버리고 description만 넘겨서
                    # F4-1이 처음부터 다시 추출하게 했다. 근데 F4-3의 description 자체가
                    # "조직 역량을 근거로" 쓰인 문장이라, 그걸 F4-1이 다시 추출하면 조직이
                    # 이미 가진 역량이 순환적으로 재검출돼 역량전이가능성이 자주 0점으로
                    # 나오는 문제가 있었다(2026-08-10 확인). 지금은 F4-1·F4-3이 같은
                    # 표준역량 32개 축을 쓰므로, F4-3의 판단을 버리지 않고 그대로 재사용한다
                    # — F4-1을 다시 태우지 않고 F4-1의 출력 형태만 그대로 흉내낸다.
                    target_tab = _find_empty_idea_tab()
                    st.session_state.setdefault("idea_tab_data", {})[target_tab] = {
                        "industry": c.get("target_industry", ""),
                        "market": c.get("target_market", ""),
                        "problem": c["description"],
                        "customer": c.get("target_customer", ""),
                    }
                    # 위젯이 이번 실행에서 안 그려진 탭은 key가 지워져 있을 수 있는데,
                    # 혹시 남아있는 값이 있으면 value=보다 그게 우선시될 수 있으니 같이 지운다.
                    for field in ("industry", "market", "problem", "customer"):
                        st.session_state.pop(f"idea_{field}_{target_tab}", None)

                    cand_ids = c.get("required_capability_ids") or []
                    cand_names = c.get("required_capability_names") or []
                    st.session_state[f"idea_req_{target_tab}"] = {
                        "input": {
                            "industry": c.get("target_industry", ""),
                            "problem": c["description"],
                            "market": c.get("target_market", ""),
                            "customer": c.get("target_customer", ""),
                        },
                        "target_industry": c.get("target_industry", ""),
                        "required_capabilities": [
                            {"capability_id": cid, "name": name, "confidence": None, "source": "f4-3"}
                            for cid, name in zip(cand_ids, cand_names)
                        ],
                        "required_capability_ids": cand_ids,
                        "required_capability_names": cand_names,
                        # F4-3은 required_capability_ids를 32개 표준역량 enum으로만 답하므로
                        # (스키마 제약) 목록 밖 역량 개념이 애초에 없다 — 항상 빈 리스트.
                        "unmatched_capabilities": [],
                        "confidence": "high",
                        "used_llm_fallback": False,
                        "llm_rationale": None,
                    }
                    # 새로 채웠으니 이전 채점(캐시)이 있었다면 버려서 다시 계산하게 한다.
                    st.session_state.pop(f"idea_score_{target_tab}", None)
                    st.session_state.pop(f"idea_judgment_{target_tab}", None)
                    # 채점은 사용자가 '적합도 판단하기'를 누른 뒤에 시작한다 — 이 탭을
                    # 전에 채점한 적이 있어도 자동으로 다시 돌지 않게 요청 표시를 지운다.
                    st.session_state.pop(f"idea_run_{target_tab}", None)

                    st.session_state["idea_active_tab"] = target_tab
                    st.session_state.tab = "f2"
                    st.rerun()

        if st.button("다시 제안받기", key="regen_new_cands", type="primary"):
            st.session_state.pop(key, None)
            st.rerun()


def missing_cap_names(sel) -> list:
    """9단계 부족 역량 — F3-1 차집합 결과(F3-2가 계산)를 사람이 읽는 이름으로.

    F3-2의 matched/missing은 'CAP_3D' 같은 표준역량ID다. 그대로 F5 프롬프트나
    화면에 넘기면 사용자가 못 읽으므로 cap_names()로 변환한다.
    """
    ids = list(sel["missing"]) if sel["missing"] is not None else []
    if f3_2 and ids:
        return f3_2.cap_names(ids)
    if ids:
        return [str(i) for i in ids]
    return []


def _safe_cell(sel, key, default=None):
    """Series/DataFrame 셀의 NaN을 F5 입력에서 제거한다."""
    value = sel.get(key, default)
    if value is None:
        return default
    try:
        if not isinstance(value, (dict, list, tuple)) and pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    return value


def _build_f5_category_details(
    sel, org_ctx: dict, diagnosis_mode: str = "capability_recommendation"
) -> dict:
    """점수 엔진이 만든 17개 세부점수를 F5 읽기 전용 입력으로 조립한다.

    조직계열은 F3-2가 반환한 sub_scores를 그대로 사용하고, 시장계열은
    F2-5의 실제 산식을 다시 호출해 DB 부분과 F 사업화 부분을 분리한다.
    data_status는 F5가 이 값의 존재 여부만 보고 파이썬에서 결정한다.
    """
    raw_org = _safe_cell(sel, "sub_scores", {})
    raw_org = raw_org if isinstance(raw_org, dict) else {}
    org_sub_caps = f3_2.SUB_CAPS if f3_2 else {}
    details = {
        category: {
            subitem: {"score": score, "max_score": org_sub_caps[category][subitem]}
            for subitem, score in subitems.items()
        }
        for category, subitems in raw_org.items()
        if category in org_sub_caps and isinstance(subitems, dict)
    }

    transfer_bonus = _safe_cell(sel, "역량전이보너스", {})
    transfer_bonus = transfer_bonus if isinstance(transfer_bonus, dict) else {}
    if transfer_bonus.get("applied"):
        subitem = transfer_bonus.get("subitem", "A입력유형전이")
        transfer_detail = details.get("역량전이가능성", {}).get(subitem)
        if isinstance(transfer_detail, dict):
            items = ", ".join(transfer_bonus.get("items") or [])
            transfer_detail["evidence"] = (
                "표준 목록 밖 역량에 대한 임시 보너스"
                + (f": {items}" if items else "")
                + "; 표준 점수와 구분해 해석"
            )

    market_size = _safe_cell(sel, "시장규모_억달러")
    barrier = _safe_cell(sel, "진입장벽_등급")
    has_business = bool(org_ctx.get("has_commercialization_experience", False))
    db_barrier = f2_5.score_entry_barrier(barrier, False) if barrier is not None else None
    db_viability = (
        f2_5.score_business_viability(market_size, False)
        if market_size is not None
        else None
    )
    market_estimate = _safe_cell(sel, "LLM시장추정", {})
    market_estimate = market_estimate if isinstance(market_estimate, dict) else {}
    if diagnosis_mode == "idea_fit":
        market_evidence = " / ".join(
            part for part in (
                f"LLM 추정 시장규모: {market_estimate.get('market_size_usd')}억달러"
                if market_estimate.get("market_size_usd") is not None else "",
                f"LLM 추정 진입장벽: {market_estimate.get('entry_barrier')}"
                if market_estimate.get("entry_barrier") else "",
                f"추정 근거: {market_estimate.get('rationale')}"
                if market_estimate.get("rationale") else "",
            ) if part
        )
    else:
        market_evidence = " / ".join(
            part for part in (
                f"DB 시장규모: {market_size}억달러" if market_size is not None else "",
                f"DB 진입장벽: {barrier}" if barrier else "",
                str(_safe_cell(sel, "기업사례_요약", "") or ""),
            ) if part
        )

    def market_detail(score, maximum):
        return {"score": score, "max_score": maximum, "evidence": market_evidence}

    details.update({
        "시장성": {
            "DB시장규모": market_detail(_safe_cell(sel, "시장성"), 15)
        },
        "경쟁강도": {
            "DB경쟁환경": market_detail(_safe_cell(sel, "경쟁강도"), 10)
        },
        "진입장벽": {
            "DB진입장벽": market_detail(db_barrier, 5),
            "F사업화역량": market_detail(5 if has_business else 0, 5),
        },
        "사업성": {
            "DB시장규모": market_detail(db_viability, 6),
            "F사업화역량": market_detail(4 if has_business else 0, 4),
        },
    })
    return details


def _build_f5_idea_context(sel) -> dict:
    typed = _safe_cell(sel, "아이디어입력", {})
    typed = typed if isinstance(typed, dict) else {}
    market_estimate = _safe_cell(sel, "LLM시장추정", {})
    market_estimate = market_estimate if isinstance(market_estimate, dict) else {}
    return {
        "idea_id": str(_safe_cell(sel, "아이디어ID", "")),
        "name": str(_safe_cell(sel, "아이디어명", "")),
        "description": str(_safe_cell(sel, "설명", "")),
        "industry": typed.get("industry") or _safe_cell(sel, "산업분류"),
        "target_market": typed.get("market") or _safe_cell(sel, "시장규모"),
        "target_customer": typed.get("customer"),
        "problem": typed.get("problem") or _safe_cell(sel, "설명"),
        "required_capabilities": _safe_cell(sel, "필요역량태그", ""),
        "market_size": _safe_cell(sel, "시장규모"),
        "market_size_usd_100m": _safe_cell(sel, "시장규모_억달러"),
        "entry_barrier": _safe_cell(sel, "진입장벽수준")
        or _safe_cell(sel, "진입장벽_등급"),
        "company_cases": _safe_cell(sel, "실제기업사례"),
        "company_case_summary": _safe_cell(sel, "기업사례_요약"),
        "market_estimate_is_llm_generated": bool(market_estimate.get("is_estimated")),
        "market_estimate_rationale": market_estimate.get("rationale"),
    }


def _nonfull_score_categories(sel) -> list[str]:
    """명시적 missing이 없어도 리포트가 필요한 비만점 항목을 찾는다."""

    result = []
    for category, maximum, _group in ITEMS:
        value = _safe_cell(sel, category)
        if value is None:
            continue
        try:
            if float(value) < float(maximum) - 0.01:
                result.append(category)
        except (TypeError, ValueError):
            continue
    return result


def render_gap_report(sel, profile: dict | None, org_ctx: dict,
                      key_prefix: str = "gap",
                      diagnosis_mode: str = "capability_recommendation") -> None:
    """9~11단계 — 새 F5 구조화 리포트를 클릭형 화면으로 표시한다."""
    if not f5 or not profile:
        st.markdown("""
<div class="note-card wait">
  <p class="section-title">역량 갭 리포트 — 준비중</p>
  <p>갭 리포트 생성 연결 대기 (조직 데이터 업로드 필요)</p>
</div>""", unsafe_allow_html=True)
        return

    missing_caps = missing_cap_names(sel)
    nonfull_categories = _nonfull_score_categories(sel)
    idea_id = str(sel["아이디어ID"])
    # F5 문장 생성 계약이 바뀌면 키도 바꿔 이전 세션의 낡은 리포트를 재사용하지 않는다.
    report_key = f"{key_prefix}_report_f5_luna_v8_{diagnosis_mode}_{idea_id}"
    gap_report, gap_error = st.session_state.get(report_key, (None, None))

    if not missing_caps and not nonfull_categories:
        st.markdown("""
<div class="note-card">
  <p class="section-title">역량 갭 리포트</p>
  <p>모든 평가항목이 만점이므로 현재 추가로 보완할 항목이 없습니다.</p>
</div>""", unsafe_allow_html=True)
        return

    target = (
        100.0
        if diagnosis_mode == "idea_fit"
        else float(sel["배점합"]) if sel["배점합"] else 100.0
    )
    if gap_report is None and gap_error is None:
        if missing_caps:
            focus_text = (
                f"부족 역량 {len(missing_caps)}개 · {', '.join(missing_caps)}"
            )
        else:
            focus_text = (
                f"비만점 평가항목 {len(nonfull_categories)}개 · "
                f"{', '.join(nonfull_categories)} · 점수 결과에서 보완과제를 자동 추출합니다."
            )
        st.markdown(f"""
<div class="note-card wait">
  <p class="section-title">역량 갭 리포트 · 보완 로드맵</p>
  <p>{html.escape(focus_text)}</p>
  <p class="csub">현재 {sel['획득']:.1f}점 / 목표 {target:.0f}점 기준으로
    보완전략(Build·Buy·Partner·Hire)과 단기·중기·장기 로드맵을 생성합니다.</p>
</div>""", unsafe_allow_html=True)
        # 생성 중에는 버튼을 '갭 리포트 생성 중…'으로 바꿔 끼운다. st.empty()에
        # 다시 그리면 버튼이 있던 자리가 그대로 교체돼서, 버튼과 진행표시가
        # 위아래로 같이 보이지 않는다(중복으로 눌리는 것도 막힌다).
        gen_slot = st.empty()
        if gen_slot.button("AI 갭 리포트 생성", key=f"gen_{report_key}"):
            idea_context = _build_f5_idea_context(sel)
            url = _safe_cell(sel, "출처링크")
            sources = [{
                "source_id": f"idea_{idea_id}",
                "content": " / ".join(
                    f"{key}: {value}" for key, value in idea_context.items() if value
                ),
                "idea_id": idea_id,
                "idea_name": str(sel["아이디어명"]),
                "source_type": "idea_data",
                **({"source_url": str(url)} if url else {}),
            }]
            present = {n: _safe_cell(sel, n) for n, _, _ in ITEMS}
            present = {n: value for n, value in present.items() if value is not None}
            caps = {n: c for n, c, _ in ITEMS if n in present}
            reasons = [
                {"category": category, "reason": reason, "source": source}
                for category, reason, source in build_score_reasons(sel)
            ]
            gap_data = {
                "diagnosis_mode": diagnosis_mode,
                "scores": {n: {"score": float(v), "max_score": caps[n]}
                           for n, v in present.items()},
                "total_score": float(sel["획득"]),
                "target_score": target,
                "missing": missing_caps,
                "category_details": _build_f5_category_details(
                    sel, org_ctx, diagnosis_mode
                ),
                "organization_profile": profile,
                "score_reasons": reasons,
                "idea": idea_context,
                "organization_id": profile.get("organization_id"),
            }
            try:
                with gen_slot.container():
                    with st.spinner("갭 리포트 생성 중…"):
                        report = f5.generate_gap_report(gap_data, sources)
                st.session_state[report_key] = (report, None)
            except f5.F5Error as e:
                st.session_state[report_key] = (None, str(e))
            st.rerun()
        return

    if gap_error:
        st.markdown(f"""
<div class="note-card wait">
  <p class="section-title">역량 갭 리포트 — 생성 실패</p>
  <p class="csub">{html.escape(gap_error)}</p>
</div>""", unsafe_allow_html=True)
        if st.button("다시 시도", key=f"retry_{report_key}"):
            st.session_state.pop(report_key, None)
            st.rerun()
        return

    view = f5.build_dashboard_view(gap_report)
    strategy_labels = {
        "build": "자체 개발(Build)",
        "buy": "외부 도입(Buy)",
        "partner": "제휴·협력(Partner)",
        "hire": "전문인력 확보(Hire)",
    }
    applicability_labels = {
        "recommended": "권장",
        "conditional": "조건부",
        "not_applicable": "해당 없음",
    }
    phase_labels = {"short_term": "단기", "mid_term": "중기", "long_term": "장기"}

    def category_priority(category):
        score = category["score"]
        maximum = category["max_score"]
        if score is None:
            return "unscored", "미평가"
        if math.isclose(float(score), float(maximum), abs_tol=0.01):
            return "maintain", "강점 유지"
        gap_ratio = (float(maximum) - float(score)) / float(maximum)
        if gap_ratio >= 0.5:
            return "high", "우선 보완"
        if gap_ratio >= 0.25:
            return "medium", "보완 필요"
        return "low", "점검 권장"

    gaps = view["gaps"]
    primary_gap = gaps[0] if gaps else {}
    primary_strategy = primary_gap.get("priority_strategy", "partner")
    short_term = view["roadmap"].get("short_term", {})
    gap_preview = " · ".join(item["capability"] for item in gaps[:2])
    if len(gaps) > 2:
        gap_preview += f" 외 {len(gaps) - 2}개"

    # 제목과 다운로드를 한 줄에 배치한다. PDF는 이미 생성된 결과만 변환하므로
    # 버튼을 눌러도 LLM이나 외부 API를 다시 호출하지 않는다.
    subject_name = str(_safe_cell(sel, "아이디어명", "진단대상") or "진단대상")
    filename_part = _download_filename_part(subject_name)
    pdf_export = None
    pdf_error = None
    try:
        pdf_export = _build_f5_pdf_download(gap_report, subject_name)
    except f5.F5ConfigurationError as exc:
        pdf_error = str(exc)

    st.markdown('<p class="f5-report-title">역량 갭 리포트 · 보완 로드맵</p>',
                unsafe_allow_html=True)
    if pdf_error:
        st.warning(pdf_error)
    overview_cols = st.columns(3)
    overview_cards = [
        (
            "식별된 부족 역량",
            f"{len(gaps)}개 보완 과제",
            gap_preview or "추가 보완 과제 없음",
        ),
        (
            "최우선 확보 방향",
            strategy_labels.get(primary_strategy, primary_strategy),
            primary_gap.get("capability", "현재 강점 유지"),
        ),
        (
            "첫 실행 단계",
            "단기 검증 착수",
            short_term.get("headline") or short_term.get("objective", "실행계획 확인"),
        ),
    ]
    # eq-card를 붙여 3장의 높이를 항상 같게 만든다(설명 길이가 카드마다 다르다).
    for index, (column, (label, value, subtext)) in enumerate(
            zip(overview_cols, overview_cards)):
        with column:
            # 마지막 카드 뒤에는 화살표를 붙이지 않는다.
            arrow = ('<span class="f5-kpi-arrow">&rsaquo;</span>'
                     if index < len(overview_cards) - 1 else "")
            st.markdown(
                f'<div class="f5-kpi eq-card">'
                f'<div class="f5-kpi-head"><span class="f5-kpi-num">{index + 1}</span>'
                f'<p class="f5-kpi-label">{html.escape(label)}</p></div>'
                f'<p class="f5-kpi-value">{html.escape(value)}</p>'
                f'<p class="f5-kpi-sub">{_safe_llm_text(subtext)}</p>'
                f'{arrow}</div>',
                unsafe_allow_html=True,
            )

    st.markdown(
        f'<div class="f5-judgment"><p class="f5-judgment-label">종합 판단</p>'
        f'<p class="f5-judgment-text">{_safe_llm_text(view["headline"])}</p></div>',
        unsafe_allow_html=True,
    )
    # PDF 저장·전문 보기는 종합 판단 바로 아래에 나란히 둔다. PDF는 이미 생성된
    # 결과만 변환하므로 눌러도 LLM이나 외부 API를 다시 호출하지 않는다.
    summary_key = f"f5_show_summary_{report_key}"
    # 뒤따르는 버튼 줄을 CSS로 집기 위한 표식(화면엔 안 보인다).
    st.markdown('<span class="f5-actions-anchor"></span>', unsafe_allow_html=True)
    # 폭 배분은 CSS가 '내용 폭'으로 덮어쓴다 — 세 번째 칸은 남는 자리를 받는 용도.
    btn_pdf, btn_summary, _btn_rest = st.columns([1, 1, 3])
    with btn_pdf:
        if pdf_export is not None:
            st.download_button(
                "PDF 보고서 다운로드",
                data=pdf_export,
                file_name=f"{filename_part}_역량갭리포트.pdf",
                mime="application/pdf",
                key=f"download_pdf_{report_key}",
                type="primary",
                # use_container_width는 쓰지 않는다 — 래퍼가 width:100%가 되면서
                # 칸의 내용 폭 계산이 0으로 무너져 버튼이 칸 밖으로 삐져나왔다.
                # 폭은 위 CSS(--f5-btn)가 칸과 버튼에 같이 지정한다.
                help="현재 화면의 전체 분석 내용을 한글 PDF 보고서로 저장합니다.",
            )
    with btn_summary:
        if st.button("종합분석 전문 보기", key=f"toggle_summary_{report_key}"):
            st.session_state[summary_key] = not st.session_state.get(summary_key, False)
    if st.session_state.get(summary_key):
        with st.container(border=True):
            st.write(_safe_md_text(view["summary"]))

    with st.container():
        st.markdown('<span class="f5-tabs-anchor"></span>', unsafe_allow_html=True)
        tab_categories, tab_gaps, tab_strategies, tab_roadmap = st.tabs(
            ["평가항목 분석", "부족 역량 진단", "보완전략", "실행 로드맵"]
        )

    with tab_categories:
        st.markdown('<p class="f5-tab-title">8개 평가항목 요약</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="f5-section-lead">카드에서는 핵심 결론만 확인하고, 상세 분석에서 세부항목별 판단과 보완점을 볼 수 있습니다.</p>',
            unsafe_allow_html=True,
        )
        # 손봐야 할 항목이 위로 오게 정렬한다: 우선 보완 → 보완 필요 → 점검 권장
        # → 강점 유지 → 미평가. sorted()는 안정 정렬이라 같은 등급 안에서는
        # F5가 준 원래 순서(배점 순)가 그대로 유지된다.
        priority_rank = {"high": 0, "medium": 1, "low": 2, "maintain": 3, "unscored": 4}
        ordered_categories = sorted(
            view["categories"],
            key=lambda item: priority_rank.get(category_priority(item)[0], 9),
        )
        category_cols = None
        for index, category in enumerate(ordered_categories):
            if index % 2 == 0:
                category_cols = st.columns(2)
            score_text = (
                "미평가"
                if category["score"] is None
                else f"{category['score']}/{category['max_score']}점"
            )
            priority_key, priority_label = category_priority(category)
            with category_cols[index % 2]:
                with st.container(border=True):
                    st.markdown(
                        f'<span class="f5-card-anchor f5-eq-anchor f5-accent-{priority_key}"></span>'
                        f'<div class="f5-card-head">'
                        f'<p class="f5-card-title">{html.escape(category["category"])}</p>'
                        f'<span class="f5-badge f5-priority-{priority_key}">{html.escape(priority_label)}</span>'
                        f'<span class="f5-badge f5-status-not_applicable">{html.escape(score_text)}</span>'
                        f'</div><p class="f5-card-copy">{_safe_llm_text(category["headline"])}</p>',
                        unsafe_allow_html=True,
                    )
                    with st.expander("자세히 보기"):
                        # 다른 세 탭과 같은 '라벨 : 본문' 줄글로 통일한다.
                        # (표로 두면 이 탭만 형식이 달라 보이고 좁은 칸에서 글이 눌렸다)
                        body = (f'<p class="f5-detail-lead">'
                                f'{_safe_llm_text(_brief(category, "summary"))}</p>')
                        for sub in category["subitem_analysis"]:
                            score = ("자료 없음" if sub["score"] is None
                                     else f'{sub["score"]}/{sub["max_score"]}점')
                            body += (
                                f'<p class="f5-dt-sub">{html.escape(sub["display_name"])}'
                                f'<span>{html.escape(score)}</span></p>'
                                + _f5_detail_block([
                                    ("판단", _brief(sub, "assessment")),
                                    ("보완점", _brief(sub, "improvement")),
                                ]))
                        st.markdown(body, unsafe_allow_html=True)

    with tab_gaps:
        st.markdown('<p class="f5-tab-title">우선 보완 과제</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="f5-section-lead">원인과 사업상 영향은 접어두고, 우선 확보 방향과 핵심 판단부터 보여줍니다.</p>',
            unsafe_allow_html=True,
        )
        for gap in gaps:
            strategy_type = gap["priority_strategy"]
            with st.container(border=True):
                st.markdown(
                    f'<span class="f5-card-anchor f5-accent-teal"></span>'
                    f'<div class="f5-card-head">'
                    f'<p class="f5-card-title">{html.escape(gap["capability"])}</p>'
                    f'<span class="f5-badge f5-strategy-{strategy_type}">'
                    f'우선 · {html.escape(strategy_labels[strategy_type])}</span>'
                    f'</div><p class="f5-card-copy">{_safe_llm_text(gap["headline"])}</p>',
                    unsafe_allow_html=True,
                )
                with st.expander("자세히 보기"):
                    st.markdown(_f5_detail_block([
                        ("원인", _brief(gap, "cause")),
                        ("사업상 영향", _brief(gap, "impact")),
                        ("선정 이유", _brief(gap, "priority_rationale")),
                    ]), unsafe_allow_html=True)

    with tab_strategies:
        priority_by_gap = {gap["capability"]: gap["priority_strategy"] for gap in gaps}
        for strategy_group in view["strategies"]:
            st.markdown(f'<p class="f5-tab-title">{html.escape(strategy_group["capability"])}</p>',
                        unsafe_allow_html=True)
            strategy_cols = st.columns(2)
            for index, item in enumerate(strategy_group["items"]):
                strategy_type = item["strategy_type"]
                applicability = item["applicability"]
                is_priority = priority_by_gap.get(strategy_group["capability"]) == strategy_type
                priority_badge = (
                    '<span class="f5-badge f5-status-recommended">최우선</span>'
                    if is_priority else ""
                )
                # 권장 + 최우선이 함께 붙은 전략은 테두리로 강조한다(CSS f5-card-featured).
                featured = " f5-card-featured" if (is_priority and applicability == "recommended") else ""
                with strategy_cols[index % 2]:
                    with st.container(border=True):
                        st.markdown(
                            f'<span class="f5-card-anchor f5-eq-anchor f5-accent-teal{featured}"></span>'
                            f'<div class="f5-card-head">'
                            f'<p class="f5-card-title">{html.escape(strategy_labels[strategy_type])}</p>'
                            f'<span class="f5-badge f5-strategy-{strategy_type}">{strategy_type.title()}</span>'
                            f'<span class="f5-badge f5-status-{applicability}">'
                            f'{html.escape(applicability_labels[applicability])}</span>{priority_badge}'
                            f'</div><p class="f5-card-copy">{_safe_llm_text(item["headline"])}</p>',
                            unsafe_allow_html=True,
                        )
                        with st.expander("자세히 보기"):
                            st.markdown(_f5_detail_block([
                                ("실행안", _brief(item, "action")),
                                ("판단 이유", _brief(item, "rationale")),
                            ]), unsafe_allow_html=True)
            st.divider()

    with tab_roadmap:
        st.markdown('<p class="f5-tab-title">실행 로드맵</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="f5-section-lead">기간을 임의로 단정하지 않고 단기·중기·장기의 목표, 실행항목과 완료기준을 구분합니다.</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="f5-timeline">'
            '<div class="f5-timeline-step f5-timeline-short"><span class="f5-timeline-dot"></span>'
            '<span class="f5-timeline-label">단기</span><span class="f5-timeline-copy">검증 과제</span></div>'
            '<div class="f5-timeline-step f5-timeline-mid"><span class="f5-timeline-dot"></span>'
            '<span class="f5-timeline-label">중기</span><span class="f5-timeline-copy">파일럿 운영</span></div>'
            '<div class="f5-timeline-step f5-timeline-long"><span class="f5-timeline-dot"></span>'
            '<span class="f5-timeline-label">장기</span><span class="f5-timeline-copy">정식 확장</span></div>'
            '</div>',
            unsafe_allow_html=True,
        )
        roadmap_cols = st.columns(3)
        for column, (phase, label) in zip(roadmap_cols, phase_labels.items()):
            node = view["roadmap"].get(phase)
            if not node:
                continue
            with column:
                with st.container(border=True):
                    st.markdown(
                        f'<span class="f5-card-anchor f5-eq-anchor f5-roadmap-anchor '
                        f'f5-accent-{phase.replace("_term", "")}"></span>'
                        f'<p class="f5-phase f5-phase-{phase.replace("_term", "")}">{html.escape(label)}</p>'
                        f'<p class="f5-card-title">{_safe_llm_text(node["headline"])}</p>',
                        unsafe_allow_html=True,
                    )
                    with st.expander("자세히 보기"):
                        actions = node.get("actions_brief") or node["actions"]
                        criteria = (node.get("completion_criteria_brief")
                                    or node["completion_criteria"])
                        st.markdown(
                            _f5_detail_block([("목표", _brief(node, "objective"))])
                            + f'<div class="f5-dt-row"><p class="f5-dt-label">실행 항목</p>'
                            + "".join(f'<p class="f5-dt-li">{_safe_llm_text(a)}</p>'
                                      for a in actions)
                            + f'</div><div class="f5-dt-row"><p class="f5-dt-label">완료 기준</p>'
                            + "".join(f'<p class="f5-dt-li">{_safe_llm_text(c)}</p>'
                                      for c in criteria)
                            + "</div>",
                            unsafe_allow_html=True)

    # 바로 위 카드에 붙어 보여서 한 칸 띄운다(표식은 화면에 안 보인다).
    st.markdown('<span class="f5-regen-anchor"></span>', unsafe_allow_html=True)
    if st.button("다시 생성", key=f"regen_{report_key}"):
        st.session_state.pop(report_key, None)
        st.rerun()


def screen_dashboard():
    profile = load_org_profile()
    org_ctx = (profile or {}).get("org_context", {})
    # 업로드 프로필의 보유수준을 F3-2에 주입한 뒤 채점한다 (캐시 밖에서 매번)
    org_levels = apply_org_levels(profile)
    db = load_scored_db(org_ctx, org_levels)

    # ── 헤더: 로고+타이틀(왼쪽) / 탭(오른쪽) ──
    # 탭 쪽을 넓게 준다. 창을 최대화하지 않으면 오른쪽 칸이 두 버튼의 글자 폭보다
    # 좁아져서 버튼끼리 겹쳐 보였다(2026-08-09). 폭 배분 + CSS(자동 폭 컬럼 +
    # 화면 폭별 글씨 축소) 두 가지로 같이 막는다.
    head_l, head_r = st.columns([1.85, 2.15], vertical_alignment="center")
    with head_l:
        st.markdown(f"""
<div class="brand">
  <a class="brand-logo" href="?home=1" target="_self" title="처음 화면으로"></a>
  <div><h1>신사업 진단 AI</h1><p>써니C 프로젝트</p></div>
</div>""", unsafe_allow_html=True)
    with head_r:
        t1, t2 = st.columns([1, 1.25])
        with t1:
            if st.button("역량 기반 추천형", key="tab1",
                         type="primary" if st.session_state.tab == "f1" else "secondary"):
                st.session_state.tab = "f1"
                st.rerun()
        with t2:
            if st.button("아이디어 적합도 판단형", key="tab2",
                         type="primary" if st.session_state.tab == "f2" else "secondary"):
                st.session_state.tab = "f2"
                st.rerun()

    st.markdown('<hr class="top-line">', unsafe_allow_html=True)

    # ── 조직 프로필 요약 (F1) ──
    render_org_summary(profile)

    if st.session_state.tab == "f2":
        render_idea_tab(db, profile, org_ctx)
        return

    # ── 8단계 실행 가능성 평가 (F3-3) → 통과한 후보만 Top-3 ──
    scanned = feasibility_pass(db)
    dropped = scanned[~scanned["feasible"]]
    passed = scanned[scanned["feasible"]]
    if passed.empty:          # 전부 탈락하면 필터를 알리고 원본으로 보여준다
        passed = scanned
    results = retrieve_business_candidates(passed, "", top_k=3)
    picked = min(st.session_state.picked, len(results) - 1)

    if f3_3 and not dropped.empty:
        with st.expander("실행 가능성 필터 확인"):
            st.dataframe(
                dropped[["아이디어ID", "아이디어명", "총점", "drop_reason"]]
                .rename(columns={"drop_reason": "제외 사유"}),
                hide_index=True, use_container_width=True)
            st.caption(f"전체 {len(scanned)}건 중 {len(dropped)}건 제외")

    cols = st.columns(len(results))
    for i, (col, (_, row)) in enumerate(zip(cols, results.iterrows())):
        with col:
            st.markdown(candidate_card(row, i + 1, i == picked), unsafe_allow_html=True)
            # 카드 전체를 덮는 투명 버튼 (CSS가 opacity:0으로 카드 위에 겹친다).
            # 링크가 아니라 버튼이라 rerun만 일어나고 세션이 유지된다.
            if st.button(f"{i + 1}위 선택", key=f"pick_{i}"):
                st.session_state.picked = i
                st.rerun()

    sel = results.iloc[picked]

    # ── 스코어카드 ──
    # 제목 / 버튼 / 표를 각각 따로 그린다. 한 markdown에 묶으면 'DB 원문보기'를
    # 제목 텍스트 바로 옆에 붙일 수 없다(버튼은 별도 요소라 같은 문단에 못 들어간다).
    # CSS가 이 세로 블록을 줄바꿈되는 flex 행으로 만들어 제목·버튼을 한 줄에 놓고,
    # 표는 width:100%로 다음 줄로 흘린다.
    with st.container(border=True):
        st.markdown('<span class="scorecard-marker" style="display:none"></span>',
                    unsafe_allow_html=True)
        st.markdown(
            f'<p class="sc-title">{html.escape(str(sel["아이디어명"]))}</p>',
            unsafe_allow_html=True)
        if st.button("DB 원문보기 ›", key=f"dbsrc_{sel['아이디어ID']}"):
            _db_source_dialog(sel)
        st.markdown(f"""
<div class="sc-body">
{scorecard_table(sel)}
<p class="csub" style="margin-top:12px">
  100점 환산 <b>{sel['총점']:.1f}점</b>
  {'' if sel['배점합'] >= 100 else ' · —로 표시된 항목은 평가 대상이 아니어서 분모에서 제외되었습니다.'}
</p>
</div>""", unsafe_allow_html=True)

    # 1. 왼쪽 그래프 데이터 준비
    rad = [(n, m) for n, m, _ in ITEMS if sel[n] is not None and not pd.isna(sel[n])]
    svg_code = radar_svg([sel[n] for n, _ in rad], [n for n, _ in rad], [m for _, m in rad])

    # 2. 오른쪽 텍스트 데이터 준비
    n_match, n_miss = len(sel["matched"]), len(sel["missing"])
    reasons = build_score_reasons(sel)
    reason_html = "".join(
        f'<li><b>{html.escape(name)}</b> — {html.escape(text)}'
        f'<span class="evidence-tag">{html.escape(tag)}</span></li>'
        for name, text, tag in reasons)

    # 3. 양쪽을 하나로 묶어 무조건 높이가 같게 만드는 Flex 레이아웃 렌더링
    st.markdown(f"""
    <div style="display: flex; gap: 16px; align-items: stretch; margin-bottom: 16px; width: 100%;">
      
      <!-- 왼쪽 상자 (그래프) -->
      <div class="canvaswrap" style="flex: 1; margin: 0; display: flex; flex-direction: column;">
        <p class="section-title">{len(rad)}개 항목 점수 분포</p>
        <div style="flex: 1; display: flex; align-items: center; justify-content: center; min-height: 250px;">
          <!-- 🚨 핵심 수정: 그래프가 260px 이상 폭주하지 못하도록 족쇄 래퍼로 가둡니다 -->
          <div style="width: 100%; max-width: 260px; max-height: 260px; display: flex; justify-content: center;">
            {svg_code}
          </div>
        </div>
      </div>

      <!-- 오른쪽 상자 (텍스트) -->
      <div class="canvaswrap" style="flex: 1; margin: 0; display: flex; flex-direction: column;">
        <p class="section-title">점수 산출 근거</p>
        <p class="csub" style="margin-bottom:10px">
          요구역량 {n_match + n_miss}개 중 보유 {n_match} · 부족 {n_miss}
        </p>
        <ul class="reason-list">{reason_html}</ul>
      </div>

    </div>
    """, unsafe_allow_html=True)
    
    render_new_candidates(results, profile, db)
    with st.container(border=True):
        st.markdown('<span class="f5-report-shell-anchor"></span>', unsafe_allow_html=True)
        render_gap_report(sel, profile, org_ctx)

    st.write("")
    if st.button("← 처음으로", key="back_ob", type="secondary"):
        go(0)


# ════════════════════════════════════════════════════════════
# 라우팅
# ════════════════════════════════════════════════════════════
def go(n: int):
    st.session_state.screen = max(0, min(5, n))
    st.rerun()


def main():
    st.session_state.setdefault("screen", 0)
    st.session_state.setdefault("files", {})
    st.session_state.setdefault("uploaded", {})   # key → bytes (F1 파싱용)
    st.session_state.setdefault("picked", 0)
    st.session_state.setdefault("tab", "f1")

    # 링크(<a>) 클릭은 페이지가 새로 로드되면서 세션이 초기화되므로,
    # 어떤 화면/어떤 후보였는지를 쿼리 파라미터로 넘겨받아 복원한다.
    qp = st.query_params
    if qp.get("dev") == "1":                    # ?dev=1 → samples/로 개발용 실행
        st.session_state["dev_samples"] = True
    if qp.get("clear"):                         # 업로드 칩의 ✕ 클릭
        key = qp["clear"]
        st.session_state.get("uploaded", {}).pop(key, None)
        st.session_state.get("files", {}).pop(key, None)
        st.session_state.pop("profile", None)   # 파일이 바뀌면 프로필도 다시 만든다
    if qp.get("home"):
        st.query_params.clear()
        st.session_state.screen = 0
    elif qp.get("screen") is not None or qp.get("pick") is not None:
        for key, name in (("screen", "screen"), ("pick", "picked")):
            if qp.get(key) is not None:
                try:
                    st.session_state[name] = int(qp[key])
                except ValueError:
                    pass
        st.query_params.clear()

    screen = st.session_state.screen
    step_cfg = UPLOAD_STEPS.get(screen)
    # 업로드 여부는 위젯 상태가 아니라 우리가 보관한 bytes로 판단한다.
    # 위젯 상태는 화면을 벗어나면 Streamlit이 버리기 때문이다.
    inject_css(
        screen,
        step_cfg["title"] if step_cfg else "",
        uploaded=bool(step_cfg and st.session_state.get("uploaded", {}).get(step_cfg["key"])),
    )

    if screen == 0:
        screen_intro()
    elif screen in (1, 2, 3):
        screen_upload(screen)
    elif screen == 4:
        screen_done()
    else:
        screen_dashboard()


if __name__ == "__main__":
    main()
