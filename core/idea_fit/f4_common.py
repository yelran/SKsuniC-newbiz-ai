import importlib.util
import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

BASE = Path(__file__).parent


F1_SEARCH_PATHS = [
    Path(os.environ["F1_PATH"]) if os.environ.get("F1_PATH") else None,
    BASE / "F1.py",
    BASE.parent / "upload" / "F1.py",
    BASE.parent / "app" / "upload" / "F1.py",
    Path.home() / "Desktop" / "app" / "upload" / "F1.py",
]


@lru_cache(maxsize=1)
def load_f1():
    tried = []
    for p in F1_SEARCH_PATHS:
        if p is None:
            continue
        tried.append(str(p))
        if p.exists():
            spec = importlib.util.spec_from_file_location("f1", p)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError(
        "F1.py를 찾을 수 없습니다. 다음 경로를 확인했습니다:\n  - "
        + "\n  - ".join(tried)
        + "\n환경변수 F1_PATH로 직접 지정할 수 있습니다."
    )


# ─────────────────────────────────────────────────────────────
# 요구역량 축 — 표준역량 32개
# ─────────────────────────────────────────────────────────────
CAPABILITY_CHOICES = load_f1().load_standard_capability_choices()  # [{"id","name","desc"}, ...] 32개
if not CAPABILITY_CHOICES:
    CAPABILITY_CHOICES = [
        {"id": c["capability_id"], "name": c["name"], "desc": ""}
        for c in load_f1().CAPABILITY_DEFINITIONS
    ]

CAP_ID_VALUES = tuple(c["id"] for c in CAPABILITY_CHOICES)
CAP_NAME_BY_ID = {c["id"]: c["name"] for c in CAPABILITY_CHOICES}
CapabilityIdLiteral = Literal[CAP_ID_VALUES]  # OpenAI structured output enum 제약용

CAPABILITY_MENU = "\n".join(
    f"- {c['id']}: {c['name']}" + (f" — {c['desc'][:60]}" if c.get("desc") else "")
    for c in CAPABILITY_CHOICES
)


# ---------------------------------------------------------------
# OpenAI 클라이언트
# ---------------------------------------------------------------
@lru_cache(maxsize=1)
def get_openai_client():
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


# ---------------------------------------------------------------
# 프롬프트용 텍스트
# ---------------------------------------------------------------
def _capability_line(cap: dict, evidence_detail: dict) -> str:
    ids = cap.get("evidence_ids") or []          # 키가 없어도 죽지 않게
    if not ids:
        return f"- {cap.get('capability_id')} {cap.get('name')}: 근거 없음(미보유)"
    examples = []
    for eid in ids[:2]:
        ev = evidence_detail.get(eid, {})
        label = ev.get("title") or ev.get("role") or eid
        examples.append(f"{eid}({str(label)[:30]})")
    more = f" 외 {len(ids) - 2}건" if len(ids) > 2 else ""
    level = cap.get("level")
    level_str = f" [수준 {level}]" if level is not None else ""
    return f"- {cap.get('capability_id')} {cap.get('name')}{level_str}: {', '.join(examples)}{more}"


def build_org_profile_text(org_profile: dict) -> str:
    """조직 역량 프로필을 프롬프트에 넣을 텍스트로. org_context는 실제 있는 키만 쓴다."""
    lines = [
        _capability_line(c, org_profile.get("evidence", {}))
        for c in org_profile.get("capabilities", [])
    ]
    ctx = org_profile.get("org_context", {}) or {}
    ctx_items = [
        ("도메인 전문가 보유", ctx.get("has_domain_expert")),
        ("추론속도 기준 충족", ctx.get("inference_speed_ok")),
        ("파트너 네트워크 보유", ctx.get("has_partner_network")),
        ("상용화 경험", ctx.get("has_commercialization_experience")),
        ("보유 특허 수", ctx.get("patent_count")),
        ("특허 적용 산업", ctx.get("patent_applied_industries")),
    ]
    ctx_line = "조직 특성: " + ", ".join(
        f"{k}={v}" for k, v in ctx_items if v is not None
    )
    return "\n".join(lines) + "\n" + ctx_line


def build_candidates_text(candidates: list, limit: int) -> str:
    lines = []
    for c in candidates[:limit]:
        scores = c.get("scores", {})
        score_str = ", ".join(f"{k}={v}" for k, v in scores.items()) if scores else "점수 없음"
        lines.append(
            f"- id={c.get('id')} name={c.get('name')} total_score={c.get('total_score')}\n"
            f"  세부점수: {score_str}\n"
            f"  matched_capabilities={c.get('matched', [])} "
            f"missing_capabilities={c.get('missing', [])}"
        )
    return "\n".join(lines)


def valid_evidence_ids(org_profile: dict) -> set:
    """환각 검증용 — 최상위 evidence_ids가 없으면 evidence dict의 키로 대체."""
    ids = org_profile.get("evidence_ids")
    if ids:
        return set(ids)
    return set((org_profile.get("evidence") or {}).keys())


def filter_evidence_ids(cited: list, valid: set) -> tuple[list, list]:
    """(통과한 것, 걸러낸 것)"""
    kept = [e for e in cited if e in valid]
    dropped = [e for e in cited if e not in valid]
    return kept, dropped
