from __future__ import annotations

import copy
import json
import math
import os
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from numbers import Real
from typing import Any


REPORT_SCHEMA_VERSION = "f5_gap_report_v5_7_progressive_ui"
DIAGNOSIS_MODE_RECOMMENDATION = "capability_recommendation"
DIAGNOSIS_MODE_IDEA_FIT = "idea_fit"
SCORE_SCHEMA_RECOMMENDATION_VERSION = "score_recommendation_8criteria_v1"
SCORE_SCHEMA_IDEA_FIT_VERSION = "score_idea_fit_org55_llm45_v1"

SCORE_SCHEMA_8_VERSION = SCORE_SCHEMA_RECOMMENDATION_VERSION
DEFAULT_MODEL = "gpt-5.6-luna"
MAX_OUTPUT_TOKENS = 64_000
DEFAULT_REASONING_EFFORT = "medium"
DEFAULT_TEXT_VERBOSITY = "high"
EXECUTIVE_SUMMARY_MAX_CHARS = 800
EXECUTIVE_HEADLINE_MAX_CHARS = 180
CATEGORY_SUMMARY_MAX_CHARS = 450
CATEGORY_HEADLINE_MAX_CHARS = 120
SUBITEM_ASSESSMENT_MAX_CHARS = 350
SUBITEM_IMPROVEMENT_MAX_CHARS = 1_500
PRIORITY_SUBITEM_IMPROVEMENT_MAX_CHARS = 2_000
GAP_CAUSE_MAX_CHARS = 400
GAP_IMPACT_MAX_CHARS = 400
GAP_HEADLINE_MAX_CHARS = 140
GAP_PRIORITY_RATIONALE_MAX_CHARS = 300
STRATEGY_GROUP_MAX_CHARS = 2_800
STRATEGY_HEADLINE_MAX_CHARS = 160
ROADMAP_PHASE_MAX_CHARS = 1_800
ROADMAP_HEADLINE_MAX_CHARS = 160
UNVERIFIED_ITEM_MAX_CHARS = 250
STRATEGY_TEXT_LIMITS = {
    "recommended": (450, 250),
    "conditional": (350, 200),
    "not_applicable": (150, 150),
}
ROADMAP_OBJECTIVE_MAX_CHARS = 300
ROADMAP_ACTION_MAX_CHARS = 250
ROADMAP_CRITERION_MAX_CHARS = 160
ROADMAP_MAX_ACTIONS = 3
ROADMAP_MAX_CRITERIA = 3

# ── 대시보드 전용 짧은 버전────────────────────────────────

CATEGORY_SUMMARY_BRIEF_MAX_CHARS = 200
SUBITEM_ASSESSMENT_BRIEF_MAX_CHARS = 180
SUBITEM_IMPROVEMENT_BRIEF_MAX_CHARS = 300
GAP_CAUSE_BRIEF_MAX_CHARS = 300
GAP_IMPACT_BRIEF_MAX_CHARS = 300
GAP_PRIORITY_RATIONALE_BRIEF_MAX_CHARS = 200
STRATEGY_ACTION_BRIEF_MAX_CHARS = 300
STRATEGY_RATIONALE_BRIEF_MAX_CHARS = 250
ROADMAP_OBJECTIVE_BRIEF_MAX_CHARS = 200
ROADMAP_ACTION_BRIEF_MAX_CHARS = 70
ROADMAP_CRITERION_BRIEF_MAX_CHARS = 90

BASIS_TYPES = (
    "score_input",
    "organization_data",
    "idea_data",
    "mixed",
    "llm_recommendation",
)
STRATEGY_TYPES = ("build", "buy", "partner", "hire")
ROADMAP_PHASES = ("short_term", "mid_term", "long_term")

ORGANIZATION_CATEGORIES = (
    "조직역량적합도",
    "역량전이가능성",
    "부족역량수준",
    "실행가능성",
)
LLM_ADDED_CATEGORIES = ("시장성", "경쟁강도", "진입장벽", "사업성")

CATEGORY_DEFINITIONS: OrderedDict[str, dict[str, Any]] = OrderedDict(
    {
        "조직역량적합도": {
            "max_score": 20,
            "meaning": "현재 기술과 경험을 이 아이디어에 직접 활용할 수 있는 정도",
            "subitems": OrderedDict(
                {
                    "A입력유형적합": (7, "기존 기술의 데이터 유형 활용 가능성"),
                    "A수행작업적합": (6, "기존 기술의 핵심 업무 활용 가능성"),
                    "B특허분류매칭": (4, "보유 특허의 핵심 기술 연관성"),
                    "B로드맵연계": (3, "현재 기술개발 계획과의 연계성"),
                }
            ),
        },
        "역량전이가능성": {
            "max_score": 15,
            "meaning": "기존 기술을 새로운 산업과 데이터 환경으로 이전할 수 있는 정도",
            "subitems": OrderedDict(
                {
                    "A범용역량전이": (6, "데이터 증강·합성 기술의 타 분야 활용 가능성"),
                    "A입력유형전이": (6, "기존 데이터 처리 경험의 확장 가능성"),
                    "B유휴특허": (3, "미활용 특허의 신사업 활용 가능성"),
                }
            ),
        },
        "부족역량수준": {
            "max_score": 15,
            "meaning": "아이디어 실행을 위해 새로 확보해야 하는 역량의 규모",
            "subitems": OrderedDict(
                {
                    "A미매칭역량": (5, "현재 보유하지 않은 기술역량의 수준"),
                    "B특허미커버": (3, "특허로 보호되지 않은 핵심 기술영역"),
                    "C도메인인력부재": (7, "신사업 분야 전문인력의 확보 수준"),
                }
            ),
        },
        "실행가능성": {
            "max_score": 5,
            "meaning": "현재 조직이 실제 사업 검증과 운영을 시작할 수 있는 정도",
            "subitems": OrderedDict(
                {"C도메인전문성": (5, "신사업 실행에 필요한 도메인 전문성")}
            ),
        },
        "시장성": {
            "max_score": 15,
            "meaning": "목표 시장의 규모와 성장 기회",
            "subitems": OrderedDict(
                {"DB시장규모": (15, "목표 시장의 규모와 성장 기회")}
            ),
        },
        "경쟁강도": {
            "max_score": 10,
            "meaning": "현재 시장의 경쟁 환경과 차별화 여지",
            "subitems": OrderedDict(
                {
                    "DB경쟁환경": (
                        10,
                        "실제 기업사례 또는 시장·진입장벽 매트릭스 기반 경쟁 환경",
                    )
                }
            ),
        },
        "진입장벽": {
            "max_score": 10,
            "meaning": "규제·인증·데이터 권리·운영 조건을 통과할 수 있는 정도",
            "subitems": OrderedDict(
                {
                    "DB진입장벽": (5, "시장 진입에 필요한 규제·인증·계약 대응 수준"),
                    "F사업화역량": (5, "기술을 서비스와 운영체계로 전환할 수 있는 역량"),
                }
            ),
        },
        "사업성": {
            "max_score": 10,
            "meaning": "고객 수요를 수익 모델과 지속 가능한 운영으로 연결할 수 있는 정도",
            "subitems": OrderedDict(
                {
                    "DB시장규모": (6, "시장 수요를 매출 기회로 전환할 가능성"),
                    "F사업화역량": (4, "가격·판매·고객지원 체계를 운영할 역량"),
                }
            ),
        },
    }
)


class F5Error(Exception):
    """F5 모듈의 기본 예외."""


class F5InputError(F5Error, ValueError):
    """앞 단계 입력이 F5 계약을 지키지 않을 때 발생한다."""


class F5ConfigurationError(F5Error, RuntimeError):
    """API 키나 패키지 설정이 없을 때 발생한다."""


class F5ProviderError(F5Error, RuntimeError):
    """OpenAI 연결이나 공급자 오류가 발생했을 때 사용한다."""


class F5ResponseError(F5Error, ValueError):
    """구조화 출력이 파싱되지 않을 때 사용한다."""


LLMCallable = Callable[[str, str, str], str]


def generate_gap_report(
    gap_data: Mapping[str, Any],
    reference_sources: Sequence[Mapping[str, Any]] | None = None,
    *,
    llm_callable: LLMCallable | None = None,
    model: str | None = None,
    **_legacy_options: Any,
) -> dict[str, Any]:
    
    normalized = _normalize_gap_data(gap_data)
    sources = _build_sources(
        reference_sources or [],
        normalized.get("organization_profile") or {},
        normalized.get("score_reasons") or [],
    )

    if not normalized["missing"]:
        return _build_no_gap_report(normalized, sources)

    slots = _build_slots(normalized)
    output_schema = _build_output_schema(slots)
    system_prompt = _build_system_prompt(normalized)
    user_content = _build_user_content(normalized, sources, slots)
    selected_model = (
        model
        or os.getenv("F5_OPENAI_MODEL")
        or DEFAULT_MODEL
    )

    if llm_callable is None:
        raw_response = _call_openai(
            system_prompt,
            user_content,
            selected_model,
            output_schema,
        )
    else:
        raw_response = llm_callable(system_prompt, user_content, selected_model)

    payload = _parse_json_response(raw_response)
    _assert_required_slots(payload, slots)
    report = _assemble_report(normalized, payload, sources)
    report["model"] = selected_model
    return report


def build_dashboard_view(report: Mapping[str, Any]) -> dict[str, Any]:
    """Streamlit이 바로 그릴 수 있는 구조를 반환한다."""

    return {
        "headline": report.get("executive_headline", ""),
        "summary": report.get("executive_summary", ""),
        "diagnosis_mode": report.get("diagnosis_mode"),
        "score_contract": report.get("score_contract"),
        "score_summary": copy.deepcopy(report.get("score_summary", {})),
        "categories": copy.deepcopy(report.get("category_analysis", [])),
        "gaps": copy.deepcopy(report.get("gap_analysis", [])),
        "strategies": copy.deepcopy(report.get("strategies", [])),
        "roadmap": copy.deepcopy(report.get("roadmap", {})),
        "unverified_items": list(report.get("unverified_items", [])),
        "warnings": list(report.get("warnings", [])),
    }


def export_gap_report_pdf(
    report: Mapping[str, Any], *, subject_name: str | None = None
) -> bytes:

    if not isinstance(report, Mapping):
        raise F5InputError("내보낼 report는 딕셔너리여야 합니다.")

    try:
        from datetime import date
        import hashlib
        from io import BytesIO
        from pathlib import Path
        from xml.sax.saxutils import escape

        from reportlab.lib import colors
        from reportlab.lib.colors import HexColor
        from reportlab.lib.enums import TA_CENTER, TA_RIGHT
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.graphics.shapes import Circle, Drawing, Line, Polygon, String
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.platypus import (
            CondPageBreak,
            HRFlowable,
            KeepTogether,
            PageBreak,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError as exc:
        raise F5ConfigurationError(
            "PDF 저장 기능에 필요한 reportlab이 설치되지 않았습니다. "
            "터미널에서 'python -m pip install reportlab'을 한 번 실행해 주세요."
        ) from exc

    view = build_dashboard_view(report)
    context = report.get("context") if isinstance(report.get("context"), Mapping) else {}
    subject = str(subject_name or context.get("idea_name") or "진단 대상").strip()
    generated_on = date.today()
    report_id_seed = "|".join(
        str(value or "")
        for value in (
            subject,
            context.get("organization_id"),
            report.get("diagnosis_mode"),
        )
    )
    report_id_suffix = hashlib.sha1(report_id_seed.encode("utf-8")).hexdigest()[:3].upper()
    report_id = f"RPT-{generated_on.strftime('%Y%m%d')}-{report_id_suffix}"

    windows_dir = Path(os.environ.get("WINDIR", r"C:\Windows"))
    assets_dir = Path(__file__).resolve().parents[2] / "assets"
    brand_logo_path = assets_dir / "logo.png"
    asset_fonts = assets_dir / "fonts"
    font_candidates = [
        (
            asset_fonts / "Pretendard-Regular.ttf",
            asset_fonts / "Pretendard-SemiBold.ttf",
        ),
        (
            asset_fonts / "NotoSansKR-Regular.ttf",
            asset_fonts / "NotoSansKR-Bold.ttf",
        ),
        (
            windows_dir / "Fonts" / "malgun.ttf",
            windows_dir / "Fonts" / "malgunbd.ttf",
        ),
        (
            windows_dir / "Fonts" / "NotoSansKR-VF.ttf",
            windows_dir / "Fonts" / "NotoSansKR-VF.ttf",
        ),
        (
            Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
            Path("/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"),
        ),
    ]
    font_pair = next(
        ((regular, bold) for regular, bold in font_candidates if regular.exists()),
        None,
    )
    if font_pair is None:
        raise F5ConfigurationError(
            "한글 PDF를 만들 수 있는 글꼴을 찾지 못했습니다. "
            "Windows의 맑은 고딕 또는 Noto Sans KR 글꼴을 설치해 주세요."
        )
    regular_path, bold_path = font_pair
    if not bold_path.exists():
        bold_path = regular_path

    regular_font = "F5KoreanRegular"
    bold_font = "F5KoreanBold"
    registered = set(pdfmetrics.getRegisteredFontNames())
    if regular_font not in registered:
        pdfmetrics.registerFont(TTFont(regular_font, str(regular_path)))
    if bold_font not in registered:
        pdfmetrics.registerFont(TTFont(bold_font, str(bold_path)))

    navy = HexColor("#173D39")
    teal = HexColor("#2D7C74")
    teal_dark = HexColor("#245F59")
    mint = HexColor("#E9F4F1")
    mint_soft = HexColor("#F5F9F8")
    paper = HexColor("#FBFCFC")
    text = HexColor("#263936")
    muted = HexColor("#6C7F7B")
    light_muted = HexColor("#8A9996")
    border = HexColor("#D5E3E0")
    coral = HexColor("#C96758")
    coral_soft = HexColor("#FAECE9")
    orange = HexColor("#CB853A")
    orange_soft = HexColor("#FBF1E5")
    blue = HexColor("#5078A8")
    blue_soft = HexColor("#ECF2F9")
    gray = HexColor("#7A8785")
    gray_soft = HexColor("#F0F3F2")

    styles = getSampleStyleSheet()
    body = ParagraphStyle(
        "F5PrintBody",
        parent=styles["BodyText"],
        fontName=regular_font,
        fontSize=10,
        leading=16.2,
        textColor=text,
        wordWrap="CJK",
        spaceAfter=7,
    )
    compact = ParagraphStyle(
        "F5PrintCompact",
        parent=body,
        fontSize=9.3,
        leading=14.8,
        spaceAfter=5,
    )
    small = ParagraphStyle(
        "F5PrintSmall",
        parent=body,
        fontSize=8.2,
        leading=12.8,
        textColor=muted,
        spaceAfter=4,
    )
    micro = ParagraphStyle(
        "F5PrintMicro",
        parent=small,
        fontSize=7.4,
        leading=11.2,
        textColor=light_muted,
    )
    cover_title = ParagraphStyle(
        "F5PrintCoverTitle",
        parent=body,
        fontName=bold_font,
        fontSize=28,
        leading=36,
        textColor=navy,
        spaceAfter=10,
    )
    report_cover_brand = ParagraphStyle(
        "F5ReportCoverBrand",
        parent=body,
        fontSize=7.2,
        leading=10,
        textColor=teal_dark,
        spaceAfter=0,
    )
    report_cover_english = ParagraphStyle(
        "F5ReportCoverEnglish",
        parent=body,
        fontName=bold_font,
        fontSize=11.5,
        leading=15,
        textColor=teal,
        tracking=0.4,
        spaceAfter=0,
    )
    report_cover_title = ParagraphStyle(
        "F5ReportCoverTitle",
        parent=body,
        fontName=bold_font,
        fontSize=34,
        leading=43,
        textColor=HexColor("#172825"),
        spaceAfter=0,
    )
    report_cover_subject = ParagraphStyle(
        "F5ReportCoverSubject",
        parent=body,
        fontSize=10,
        leading=15,
        textColor=muted,
        spaceAfter=0,
    )
    report_cover_meta_label = ParagraphStyle(
        "F5ReportCoverMetaLabel",
        parent=body,
        fontSize=7.4,
        leading=11,
        textColor=muted,
        spaceAfter=0,
    )
    report_cover_meta = ParagraphStyle(
        "F5ReportCoverMeta",
        parent=body,
        fontName=bold_font,
        fontSize=8.1,
        leading=12.5,
        textColor=navy,
        spaceAfter=0,
    )
    cover_subject = ParagraphStyle(
        "F5PrintCoverSubject",
        parent=body,
        fontName=bold_font,
        fontSize=15,
        leading=23,
        textColor=teal_dark,
        spaceAfter=7,
    )
    cover_subtitle = ParagraphStyle(
        "F5PrintCoverSubtitle",
        parent=body,
        fontSize=10.5,
        leading=16,
        textColor=muted,
        spaceAfter=8,
    )
    eyebrow = ParagraphStyle(
        "F5PrintEyebrow",
        parent=body,
        fontName=bold_font,
        fontSize=8.5,
        leading=12,
        textColor=teal,
        tracking=1.1,
        spaceAfter=8,
    )
    section_title = ParagraphStyle(
        "F5PrintSectionTitle",
        parent=body,
        fontName=bold_font,
        fontSize=17,
        leading=23,
        textColor=navy,
        spaceAfter=2,
    )
    section_subtitle = ParagraphStyle(
        "F5PrintSectionSubtitle",
        parent=small,
        fontSize=8.6,
        leading=13,
        spaceAfter=0,
    )
    heading = ParagraphStyle(
        "F5PrintHeading",
        parent=body,
        fontName=bold_font,
        fontSize=12,
        leading=18,
        textColor=navy,
        keepWithNext=True,
        spaceBefore=8,
        spaceAfter=5,
    )
    subheading = ParagraphStyle(
        "F5PrintSubheading",
        parent=body,
        fontName=bold_font,
        fontSize=10.3,
        leading=16,
        textColor=navy,
        keepWithNext=True,
        spaceBefore=5,
        spaceAfter=4,
    )
    label = ParagraphStyle(
        "F5PrintLabel",
        parent=body,
        fontName=bold_font,
        fontSize=9.2,
        leading=14,
        textColor=teal_dark,
        keepWithNext=True,
        spaceAfter=3,
    )
    # 테두리·배경은 boxed_note()가 Table로 그린다. ParagraphStyle의 borderPadding은
    # 높이 계산에 들어가지 않아 글씨가 상자 밖으로 넘친다(위 boxed_note 주석 참고).
    improvement_body = ParagraphStyle(
        "F5PrintImprovement",
        parent=body,
    )
    callout_title = ParagraphStyle(
        "F5PrintCalloutTitle",
        parent=body,
        fontName=bold_font,
        fontSize=10.8,
        leading=16,
        textColor=teal_dark,
        spaceAfter=5,
    )
    metric_label = ParagraphStyle(
        "F5PrintMetricLabel",
        parent=small,
        alignment=TA_CENTER,
        spaceAfter=3,
    )
    metric_value = ParagraphStyle(
        "F5PrintMetricValue",
        parent=body,
        fontName=bold_font,
        fontSize=17,
        leading=22,
        alignment=TA_CENTER,
        textColor=teal,
        spaceAfter=0,
    )
    card_label = ParagraphStyle(
        "F5PrintCardLabel",
        parent=small,
        fontName=bold_font,
        spaceAfter=3,
    )
    card_value = ParagraphStyle(
        "F5PrintCardValue",
        parent=body,
        fontName=bold_font,
        fontSize=10.3,
        leading=15,
        textColor=navy,
        spaceAfter=3,
    )
    card_sub = ParagraphStyle(
        "F5PrintCardSub",
        parent=small,
        fontSize=7.9,
        leading=12,
        spaceAfter=0,
    )
    white_badge = ParagraphStyle(
        "F5PrintWhiteBadge",
        parent=small,
        fontName=bold_font,
        alignment=TA_CENTER,
        textColor=colors.white,
        spaceAfter=0,
    )
    right_score = ParagraphStyle(
        "F5PrintRightScore",
        parent=body,
        fontName=bold_font,
        fontSize=10,
        leading=15,
        alignment=TA_RIGHT,
        textColor=navy,
        spaceAfter=0,
    )

    def safe(value: Any) -> str:
        return escape(str(value or "")).replace("\n", "<br/>")

    def p(value: Any, style: ParagraphStyle = body) -> Paragraph:
        return Paragraph(safe(value), style)

    def labeled(label_text: str, value: Any, style: ParagraphStyle = body) -> Paragraph:
        return Paragraph(
            f'<font name="{bold_font}" color="{teal_dark.hexval()}">'
            f"{safe(label_text)}</font> {safe(value)}",
            style,
        )

    def score_text(score: Any, maximum: Any) -> str:
        return "미평가" if score is None else f"{score}/{maximum}점"

    def evidence_text(node: Mapping[str, Any]) -> str:
        parts = []
        if node.get("basis_label"):
            parts.append(f"근거 구분: {node['basis_label']}")
        if node.get("score_origin_label"):
            parts.append(f"점수 출처: {node['score_origin_label']}")
        labels = [str(item) for item in (node.get("source_labels") or []) if item]
        if labels:
            parts.append("출처: " + " / ".join(labels))
        return " | ".join(parts)

    strategy_labels = {
        "build": "자체 개발(Build)",
        "buy": "외부 도입(Buy)",
        "partner": "제휴·협력(Partner)",
        "hire": "전문인력 확보(Hire)",
    }
    strategy_short = {
        "build": "Build",
        "buy": "Buy",
        "partner": "Partner",
        "hire": "Hire",
    }
    applicability_labels = {
        "recommended": "권장",
        "conditional": "조건부",
        "not_applicable": "해당 없음",
    }
    applicability_colors = {
        "recommended": teal,
        "conditional": orange,
        "not_applicable": gray,
    }
    applicability_surfaces = {
        "recommended": mint,
        "conditional": orange_soft,
        "not_applicable": gray_soft,
    }

    evidence_lookup: OrderedDict[str, str] = OrderedDict()

    def register_evidence(node: Mapping[str, Any]) -> None:
        evidence = evidence_text(node)
        if evidence and evidence not in evidence_lookup:
            evidence_lookup[evidence] = f"E{len(evidence_lookup) + 1:02d}"

    for category in view["categories"]:
        register_evidence(category)
        for subitem in category.get("subitem_analysis", []):
            register_evidence(subitem)
    for gap in view["gaps"]:
        register_evidence(gap)
    for group in view["strategies"]:
        for item in group.get("items", []):
            register_evidence(item)
    for node in view["roadmap"].values():
        if isinstance(node, Mapping):
            register_evidence(node)

    def evidence_ref(node: Mapping[str, Any]) -> str:
        return evidence_lookup.get(evidence_text(node), "")

    buffer = BytesIO()
    page_width, page_height = A4
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=22 * mm,
        bottomMargin=18 * mm,
        title=f"{subject} 역량 갭 리포트",
        author="써니C 신사업 진단 AI",
        subject="역량 갭 진단, 보완전략 및 실행 로드맵",
    )

    def draw_page(canvas, _doc):
        page_number = canvas.getPageNumber()
        canvas.saveState()

        if page_number == 1:
            canvas.setFillColor(HexColor("#DDF1EC"))
            canvas.rect(0, 0, page_width, page_height, fill=1, stroke=0)

            def cover_paragraph(value, style, x, y_top, width):
                paragraph = p(value, style)
                _, height = paragraph.wrap(width, page_height)
                paragraph.drawOn(canvas, x, y_top - height)
                return height

            def sparkle(x, y, radius):
                path = canvas.beginPath()
                path.moveTo(x, y + radius)
                path.lineTo(x + radius * 0.22, y + radius * 0.22)
                path.lineTo(x + radius, y)
                path.lineTo(x + radius * 0.22, y - radius * 0.22)
                path.lineTo(x, y - radius)
                path.lineTo(x - radius * 0.22, y - radius * 0.22)
                path.lineTo(x - radius, y)
                path.lineTo(x - radius * 0.22, y + radius * 0.22)
                path.close()
                canvas.setFillColor(colors.white)
                canvas.drawPath(path, fill=1, stroke=0)
                canvas.circle(x + radius * 0.78, y - radius * 0.76, radius * 0.12, fill=1, stroke=0)

            left = 22 * mm
            cover_width = 150 * mm
            cover_paragraph(
                "써니C 신사업 진단 AI",
                report_cover_brand,
                left,
                page_height - 30 * mm,
                80 * mm,
            )
            sparkle(left + 7 * mm, page_height - 42 * mm, 7 * mm)
            sparkle(page_width - 36 * mm, 112 * mm, 4.2 * mm)

            cover_paragraph(
                "CAPABILITY GAP REPORT",
                report_cover_english,
                left,
                page_height - 78 * mm,
                cover_width,
            )
            cover_paragraph(
                "역량 갭",
                report_cover_title,
                left,
                page_height - 88 * mm,
                90 * mm,
            )
            cover_paragraph(
                "리포트",
                report_cover_title,
                left,
                page_height - 105 * mm,
                90 * mm,
            )
            cover_paragraph(
                subject,
                report_cover_subject,
                left,
                page_height - 139 * mm,
                138 * mm,
            )

            cover_paragraph(
                "이 보고서는 다음 대상에 대해 작성되었습니다.",
                report_cover_meta_label,
                left,
                58 * mm,
                118 * mm,
            )
            cover_paragraph(
                f"{subject} ({report_id})",
                report_cover_meta,
                left,
                52 * mm,
                118 * mm,
            )
            cover_paragraph(
                f"발행일 | {generated_on.strftime('%Y년 %m월 %d일')}",
                report_cover_meta_label,
                left,
                45 * mm,
                118 * mm,
            )

            if brand_logo_path.exists():
                canvas.drawImage(
                    str(brand_logo_path),
                    page_width - 62 * mm,
                    26 * mm,
                    width=49 * mm,
                    height=34.6 * mm,
                    preserveAspectRatio=True,
                    anchor="c",
                    mask="auto",
                )
            canvas.restoreState()
            return

        canvas.setStrokeColor(border)
        canvas.setLineWidth(0.55)
        canvas.line(
            18 * mm,
            page_height - 14 * mm,
            page_width - 18 * mm,
            page_height - 14 * mm,
        )
        canvas.setFont(regular_font, 7.7)
        canvas.setFillColor(muted)
        header = subject if len(subject) <= 42 else subject[:41] + "…"
        canvas.drawString(18 * mm, page_height - 10.7 * mm, header)
        canvas.setFont(regular_font, 7.4)
        canvas.setFillColor(light_muted)
        canvas.drawString(18 * mm, 9.5 * mm, "써니C 신사업 진단 AI")
        canvas.drawRightString(
            page_width - 18 * mm,
            9.5 * mm,
            f"p. {page_number - 1}",
        )
        canvas.restoreState()

    def section_band(number: str, title_text: str, subtitle_text: str) -> Table:
        number_box = Table(
            [[p(number, white_badge)]],
            colWidths=[11 * mm],
            rowHeights=[11 * mm],
        )
        number_box.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), teal),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 1),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 1),
                    ("TOPPADDING", (0, 0), (-1, -1), 1),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
                ]
            )
        )
        band = Table(
            [[
                number_box,
                [p(title_text, section_title), p(subtitle_text, section_subtitle)],
            ]],
            colWidths=[15 * mm, doc.width - 15 * mm],
        )
        band.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LINEBELOW", (0, 0), (-1, -1), 1.1, teal),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
                ]
            )
        )
        return band

    def callout(
        title_text: str,
        value: Any,
        accent=teal,
        background=mint,
    ) -> Table:
        box = Table(
            [[[p(title_text, callout_title), p(value, body)]]],
            colWidths=[doc.width],
        )
        box.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), background),
                    ("LINEBEFORE", (0, 0), (0, -1), 4, accent),
                    ("BOX", (0, 0), (-1, -1), 0.6, border),
                    ("LEFTPADDING", (0, 0), (-1, -1), 12),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                    ("TOPPADDING", (0, 0), (-1, -1), 10),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
        return box

    def boxed_note(value: Any, style: ParagraphStyle = body) -> Table:
        """테두리 있는 본문 상자.

        ParagraphStyle의 borderPadding으로 같은 모양을 내면 글씨가 상자 밖으로
        넘친다 — ReportLab의 Paragraph.wrap()이 borderPadding을 높이에 넣지 않아
        위아래 여백(8pt씩)만큼 자리를 덜 잡기 때문이다(reportlab 4.5.1에서 실측).
        Table은 padding을 높이에 정확히 반영하므로 한 칸짜리 표로 대신한다.
        """
        box = Table([[p(value, style)]], colWidths=[doc.width])
        box.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), mint_soft),
                    ("BOX", (0, 0), (-1, -1), 0.6, border),
                    ("LEFTPADDING", (0, 0), (-1, -1), 9),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
        return box

    def score_radar(categories: Sequence[Mapping[str, Any]]) -> Table:
        """사이트의 평가항목 레이더를 PDF용 벡터 그래프로 재현한다."""

        plotted = [
            item
            for item in categories
            if isinstance(item, Mapping)
            and isinstance(item.get("score"), Real)
            and isinstance(item.get("max_score"), Real)
            and float(item["max_score"]) > 0
        ]
        if len(plotted) < 3:
            return callout(
                "8개 항목 점수 분포",
                "그래프를 그릴 수 있는 평가항목 점수가 충분하지 않습니다.",
                gray,
                gray_soft,
            )

        width = doc.width - 16 * mm
        height = 79 * mm
        center_x = width / 2
        center_y = height * 0.46
        radius = 27 * mm
        label_radius = 36 * mm
        count = len(plotted)

        def point(index: int, distance: float) -> tuple[float, float]:
            angle = math.pi / 2 - index * 2 * math.pi / count
            return (
                center_x + distance * math.cos(angle),
                center_y + distance * math.sin(angle),
            )

        drawing = Drawing(width, height)
        for fraction in (0.25, 0.5, 0.75, 1.0):
            ring_points: list[float] = []
            for index in range(count):
                ring_points.extend(point(index, radius * fraction))
            drawing.add(
                Polygon(
                    ring_points,
                    fillColor=None,
                    strokeColor=border,
                    strokeWidth=0.55,
                )
            )

        for index in range(count):
            outer_x, outer_y = point(index, radius)
            drawing.add(
                Line(
                    center_x,
                    center_y,
                    outer_x,
                    outer_y,
                    strokeColor=border,
                    strokeWidth=0.55,
                )
            )

        data_points: list[float] = []
        for index, item in enumerate(plotted):
            ratio = max(
                0.0,
                min(1.0, float(item["score"]) / float(item["max_score"])),
            )
            data_points.extend(point(index, radius * ratio))
        drawing.add(
            Polygon(
                data_points,
                fillColor=HexColor("#D4ECE8"),
                strokeColor=teal,
                strokeWidth=1.8,
            )
        )
        for index, item in enumerate(plotted):
            ratio = max(
                0.0,
                min(1.0, float(item["score"]) / float(item["max_score"])),
            )
            dot_x, dot_y = point(index, radius * ratio)
            drawing.add(Circle(dot_x, dot_y, 2.1, fillColor=teal, strokeColor=teal))

            label_x, label_y = point(index, label_radius)
            horizontal = math.cos(math.pi / 2 - index * 2 * math.pi / count)
            if horizontal > 0.28:
                anchor = "start"
            elif horizontal < -0.28:
                anchor = "end"
            else:
                anchor = "middle"
            drawing.add(
                String(
                    label_x,
                    label_y + 2,
                    str(item.get("category") or "평가항목"),
                    fontName=bold_font,
                    fontSize=8.8,
                    fillColor=text,
                    textAnchor=anchor,
                )
            )
            drawing.add(
                String(
                    label_x,
                    label_y - 10,
                    score_text(item.get("score"), item.get("max_score")),
                    fontName=regular_font,
                    fontSize=7.2,
                    fillColor=muted,
                    textAnchor=anchor,
                )
            )

        chart = Table(
            [[
                [
                    p("8개 항목 점수 분포", card_value),
                    p("각 평가항목의 배점 대비 획득 비율", card_sub),
                    Spacer(1, 3),
                    drawing,
                ]
            ]],
            colWidths=[doc.width],
        )
        chart.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                    ("BOX", (0, 0), (-1, -1), 0.8, border),
                    ("LINEABOVE", (0, 0), (-1, 0), 3, teal),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8 * mm),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8 * mm),
                    ("TOPPADDING", (0, 0), (-1, -1), 7 * mm),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4 * mm),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
        return chart

    score_summary = view["score_summary"]
    gaps = view["gaps"]
    primary_gap = gaps[0] if gaps else {}
    primary_strategy = primary_gap.get("priority_strategy", "partner")
    short_term = view["roadmap"].get("short_term", {})
    gap_preview = " · ".join(str(item.get("capability", "")) for item in gaps[:2])
    if len(gaps) > 2:
        gap_preview += f" 외 {len(gaps) - 2}개"
    if not gap_preview:
        gap_preview = "현재 강점 유지"

    metrics = Table(
        [[
            [
                p("현재 점수", metric_label),
                p(f"{score_summary.get('current_score', '—')}점", metric_value),
            ],
            [
                p("목표 점수", metric_label),
                p(f"{score_summary.get('target_score', '—')}점", metric_value),
            ],
            [
                p("보완 차이", metric_label),
                p(f"{score_summary.get('score_gap', '—')}점", metric_value),
            ],
        ]],
        colWidths=[doc.width / 3] * 3,
    )
    metrics.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), mint_soft),
                ("BOX", (0, 0), (-1, -1), 0.8, border),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, border),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 11),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 11),
            ]
        )
    )
    overview = Table(
        [[
            [
                p("식별된 부족 역량", card_label),
                p(f"{len(gaps)}개 보완 과제", card_value),
                p(gap_preview, card_sub),
            ],
            [
                p("최우선 확보 방향", card_label),
                p(strategy_labels.get(primary_strategy, primary_strategy), card_value),
                p(primary_gap.get("capability", "현재 강점 유지"), card_sub),
            ],
            [
                p("첫 실행 단계", card_label),
                p("단기 검증 착수", card_value),
                p(
                    short_term.get("headline")
                    or short_term.get("objective", "실행계획 확인"),
                    card_sub,
                ),
            ],
        ]],
        colWidths=[doc.width / 3] * 3,
    )
    overview.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                ("BOX", (0, 0), (-1, -1), 0.8, border),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, border),
                ("LINEABOVE", (0, 0), (0, 0), 3, coral),
                ("LINEABOVE", (1, 0), (1, 0), 3, blue),
                ("LINEABOVE", (2, 0), (2, 0), 3, teal),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
            ]
        )
    )

    # 표지(캔버스로 직접 그린다) 다음의 요약 면. 제목부터 레이더 그래프까지를
    # KeepTogether로 묶어, 제목이나 종합 판단이 길어져도 그래프가 다음 장으로
    # 밀려나지 않게 한다.
    story = [
        Spacer(1, 245 * mm),
        PageBreak(),
        KeepTogether([
            p("SUNNY C · NEW BUSINESS DIAGNOSTIC AI", eyebrow),
            p("역량 갭 리포트", cover_title),
            p(subject, cover_subject),
            p("부족 역량 진단 · 보완전략 · 실행 로드맵", cover_subtitle),
            HRFlowable(
                width="100%",
                thickness=1.4,
                color=teal,
                spaceBefore=5,
                spaceAfter=13,
            ),
            labeled("평가 계약", view.get("score_contract") or "미확인", small),
            labeled("생성일", date.today().strftime("%Y.%m.%d"), small),
            Spacer(1, 7),
            metrics,
            Spacer(1, 13),
            callout("종합 판단", view["headline"]),
            Spacer(1, 10),
            score_radar(view["categories"]),
        ]),
        PageBreak(),
        section_band(
            "00",
            "종합 요약",
            "핵심 보완 과제와 우선 확보 방향, 첫 실행 단계를 한눈에 정리했습니다.",
        ),
        Spacer(1, 10),
        overview,
        Spacer(1, 10),
        p(view["summary"], compact),
        Spacer(1, 14),
        section_band(
            "01",
            "평가항목 분석",
            "점수와 핵심 결론을 먼저 확인하고 세부항목별 판단과 보완점을 읽을 수 있습니다.",
        ),
        Spacer(1, 10),
    ]

    priority_design = {
        "high": ("우선 보완", coral, coral_soft),
        "medium": ("점검 권장", orange, orange_soft),
        "low": ("보완 검토", blue, blue_soft),
        "maintain": ("강점 유지", teal, mint),
        "not_scored": ("확인 필요", gray, gray_soft),
    }
    for category in view["categories"]:
        status_text, status_color, status_surface = priority_design.get(
            category.get("priority"),
            priority_design["not_scored"],
        )
        category_header = Table(
            [[
                p(category["category"], heading),
                p(status_text, white_badge),
                p(
                    score_text(category.get("score"), category.get("max_score")),
                    right_score,
                ),
            ]],
            colWidths=[doc.width * 0.61, doc.width * 0.20, doc.width * 0.19],
        )
        category_header.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (0, 0), paper),
                    ("BACKGROUND", (1, 0), (1, 0), status_color),
                    ("BACKGROUND", (2, 0), (2, 0), status_surface),
                    ("BOX", (0, 0), (-1, -1), 0.7, border),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 9),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                    ("TOPPADDING", (0, 0), (-1, -1), 7),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ]
            )
        )
        story.extend(
            [
                CondPageBreak(63 * mm),
                category_header,
                Spacer(1, 6),
                labeled("핵심 결론", category["headline"], compact),
                labeled("종합 분석", category["summary"]),
            ]
        )
        for subitem in category.get("subitem_analysis", []):
            subitem_header = Table(
                [[
                    p(subitem["display_name"], subheading),
                    p(
                        score_text(
                            subitem.get("score"),
                            subitem.get("max_score"),
                        ),
                        right_score,
                    ),
                ]],
                colWidths=[doc.width * 0.78, doc.width * 0.22],
            )
            subitem_header.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, -1), mint_soft),
                        ("LINEBEFORE", (0, 0), (0, -1), 3, status_color),
                        ("BOX", (0, 0), (-1, -1), 0.55, border),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 8),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                        ("TOPPADDING", (0, 0), (-1, -1), 5),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ]
                )
            )
            story.extend(
                [
                    CondPageBreak(42 * mm),
                    subitem_header,
                    Spacer(1, 4),
                    p("판단", label),
                    p(subitem["assessment"], compact),
                    p("보완점", label),
                    boxed_note(subitem["improvement"], improvement_body),
                    Spacer(1, 8),
                ]
            )
        story.extend(
            [
                Spacer(1, 5),
                HRFlowable(
                    width="100%",
                    thickness=0.55,
                    color=border,
                    spaceAfter=8,
                ),
            ]
        )

    story.extend(
        [
            PageBreak(),
            section_band(
                "02",
                "부족 역량 진단",
                "현재 조직과 사업 요구조건의 차이, 사업상 영향과 최우선 확보 방향을 진단합니다.",
            ),
            Spacer(1, 10),
        ]
    )
    if not gaps:
        story.append(callout("진단 결과", "추가로 식별된 부족 역량이 없습니다."))
    for index, gap in enumerate(gaps, start=1):
        strategy_type = gap["priority_strategy"]
        strategy_name = strategy_labels.get(strategy_type, strategy_type)
        header = Table(
            [[
                p(f"진단 {index:02d}", white_badge),
                p(gap["capability"], heading),
                p(strategy_short.get(strategy_type, strategy_type), white_badge),
            ]],
            colWidths=[20 * mm, doc.width - 47 * mm, 27 * mm],
        )
        header.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (0, 0), coral),
                    ("BACKGROUND", (1, 0), (1, 0), paper),
                    ("BACKGROUND", (2, 0), (2, 0), teal),
                    ("BOX", (0, 0), (-1, -1), 0.7, border),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                    ("TOPPADDING", (0, 0), (-1, -1), 7),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ]
            )
        )
        story.extend(
            [
                CondPageBreak(75 * mm),
                header,
                Spacer(1, 6),
                callout("핵심 판단", gap["headline"], coral, coral_soft),
                Spacer(1, 7),
                p("부족 원인", label),
                p(gap["cause"]),
                p("사업상 영향", label),
                p(gap["impact"]),
                callout(
                    f"우선 확보 방향 · {strategy_name}",
                    gap["priority_rationale"],
                ),
            ]
        )
        story.append(Spacer(1, 12))

    story.extend(
        [
            PageBreak(),
            section_band(
                "03",
                "부족 역량별 보완전략",
                "Build·Buy·Partner·Hire를 비교하고 권장 여부와 구체 실행조건을 제시합니다.",
            ),
            Spacer(1, 10),
        ]
    )
    strategy_groups = list(view["strategies"])
    if not strategy_groups:
        story.append(callout("전략 결과", "추가 보완전략이 없습니다."))
    for group_index, group in enumerate(strategy_groups, start=1):
        story.extend(
            [
                CondPageBreak(60 * mm),
                p(f"{group_index:02d}. {group['capability']}", section_title),
                Spacer(1, 4),
            ]
        )
        items = list(group.get("items", []))
        if items:
            summary = Table(
                [[
                    [
                        p(
                            strategy_short.get(item["strategy_type"], item["strategy_type"]),
                            card_value,
                        ),
                        p(
                            applicability_labels.get(
                                item["applicability"],
                                item["applicability"],
                            ),
                            card_sub,
                        ),
                    ]
                    for item in items
                ]],
                colWidths=[doc.width / len(items)] * len(items),
            )
            summary_style = [
                ("BOX", (0, 0), (-1, -1), 0.6, border),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, border),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
            for column, item in enumerate(items):
                applicability = item["applicability"]
                summary_style.extend(
                    [
                        (
                            "BACKGROUND",
                            (column, 0),
                            (column, 0),
                            applicability_surfaces[applicability],
                        ),
                        (
                            "LINEABOVE",
                            (column, 0),
                            (column, 0),
                            3,
                            applicability_colors[applicability],
                        ),
                    ]
                )
            summary.setStyle(TableStyle(summary_style))
            story.extend([summary, Spacer(1, 8)])
        for item_index, item in enumerate(items, start=1):
            strategy_type = item["strategy_type"]
            applicability = item["applicability"]
            strategy_name = strategy_labels.get(strategy_type, strategy_type)
            header = Table(
                [[
                    p(strategy_name, subheading),
                    p(
                        applicability_labels.get(applicability, applicability),
                        white_badge,
                    ),
                ]],
                colWidths=[doc.width * 0.82, doc.width * 0.18],
            )
            header.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (0, 0), paper),
                        (
                            "BACKGROUND",
                            (1, 0),
                            (1, 0),
                            applicability_colors[applicability],
                        ),
                        ("BOX", (0, 0), (-1, -1), 0.65, border),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 8),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                        ("TOPPADDING", (0, 0), (-1, -1), 6),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ]
                )
            )
            story.extend(
                [
                    CondPageBreak(48 * mm),
                    header,
                    Spacer(1, 4),
                    p(item["headline"], callout_title),
                    p("실행안", label),
                    p(item["action"]),
                    p("판단 이유", label),
                    p(item["rationale"], compact),
                ]
            )
            if item_index < len(items) or group_index < len(strategy_groups):
                story.append(Spacer(1, 7))
        if group_index < len(strategy_groups):
            story.append(
                HRFlowable(
                    width="100%",
                    thickness=0.7,
                    color=border,
                    spaceAfter=11,
                )
            )

    story.extend(
        [
            PageBreak(),
            section_band(
                "04",
                "실행 로드맵",
                "단기 검증에서 중기 파일럿, 장기 운영 확장으로 이어지는 실행 흐름입니다.",
            ),
            Spacer(1, 10),
        ]
    )
    phases = (
        ("short_term", "단기", coral, coral_soft),
        ("mid_term", "중기", orange, orange_soft),
        ("long_term", "장기", teal, mint),
    )
    timeline = Table(
        [[
            [
                p(phase_label, card_value),
                p(
                    (view["roadmap"].get(phase) or {}).get("headline")
                    or "계획 확인",
                    card_sub,
                ),
            ]
            for phase, phase_label, _color, _background in phases
        ]],
        colWidths=[doc.width / 3] * 3,
    )
    timeline_style = [
        ("BOX", (0, 0), (-1, -1), 0.7, border),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, border),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
    ]
    for column, (_phase, _phase_label, color, background) in enumerate(phases):
        timeline_style.extend(
            [
                ("BACKGROUND", (column, 0), (column, 0), background),
                ("LINEABOVE", (column, 0), (column, 0), 4, color),
            ]
        )
    timeline.setStyle(TableStyle(timeline_style))
    story.extend([timeline, Spacer(1, 13)])

    for phase, phase_label, color, background in phases:
        node = view["roadmap"].get(phase)
        if not node:
            continue
        header = Table(
            [[
                p(phase_label, white_badge),
                p(node["headline"], heading),
            ]],
            colWidths=[19 * mm, doc.width - 19 * mm],
        )
        header.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (0, 0), color),
                    ("BACKGROUND", (1, 0), (1, 0), paper),
                    ("BOX", (0, 0), (-1, -1), 0.7, border),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                    ("TOPPADDING", (0, 0), (-1, -1), 7),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ]
            )
        )
        story.extend(
            [
                CondPageBreak(73 * mm),
                header,
                Spacer(1, 6),
                callout("단계 목표", node["objective"], color, background),
                Spacer(1, 6),
                p("실행 항목", label),
            ]
        )
        for index, action in enumerate(node.get("actions", []), start=1):
            story.append(labeled(f"{index}.", action, compact))
        story.append(p("완료 기준", label))
        for index, criterion in enumerate(
            node.get("completion_criteria", []),
            start=1,
        ):
            story.append(labeled(f"기준 {index}", criterion, compact))
        story.append(Spacer(1, 12))

    doc.build(story, onFirstPage=draw_page, onLaterPages=draw_page)
    return buffer.getvalue()


def _normalize_gap_data(gap_data: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(gap_data, Mapping):
        raise F5InputError("gap_data는 딕셔너리여야 합니다.")

    warnings: list[str] = []
    diagnosis_mode = _normalize_diagnosis_mode(gap_data.get("diagnosis_mode"))
    score_contract = (
        SCORE_SCHEMA_IDEA_FIT_VERSION
        if diagnosis_mode == DIAGNOSIS_MODE_IDEA_FIT
        else SCORE_SCHEMA_RECOMMENDATION_VERSION
    )
    raw_scores = gap_data.get("scores") or {}
    if not isinstance(raw_scores, Mapping):
        raise F5InputError("scores는 딕셔너리여야 합니다.")

    scores: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for category, definition in CATEGORY_DEFINITIONS.items():
        raw = raw_scores.get(category)
        score: float | None
        max_score = float(definition["max_score"])
        if raw is None:
            score = None
        elif isinstance(raw, Mapping):
            score = _optional_number(raw.get("score"), f"scores.{category}.score")
            supplied_max = _optional_number(
                raw.get("max_score"), f"scores.{category}.max_score"
            )
            if supplied_max is not None and not math.isclose(
                supplied_max, max_score, abs_tol=0.01
            ):
                warnings.append(
                    f"{category} 배점 {supplied_max:g}은 확정 배점 {max_score:g}으로 교정했습니다."
                )
        else:
            score = _optional_number(raw, f"scores.{category}")
        if score is not None and not 0 <= score <= max_score:
            raise F5InputError(
                f"scores.{category} 점수는 0~{max_score:g} 범위여야 합니다."
            )
        scores[category] = {
            "score": _tidy(score),
            "max_score": _tidy(max_score),
            "status": "scored" if score is not None else "not_scored",
            "score_origin": _score_origin(diagnosis_mode, category),
        }

    details = _normalize_category_details(
        gap_data.get("category_details"), scores, diagnosis_mode
    )
    missing = _normalize_missing(gap_data.get("missing"))
    if not missing:
        missing = _derive_score_based_gaps(scores, details)
        if missing:
            warnings.append(
                "명시적 부족역량 목록이 없어 비만점 평가항목에서 보완과제를 자동 추출했습니다."
            )
    scored_values = [x["score"] for x in scores.values() if x["score"] is not None]
    calculated_total = round(sum(float(v) for v in scored_values), 2)
    calculated_target = round(
        sum(
            float(item["max_score"])
            for item in scores.values()
            if item["score"] is not None
        ),
        2,
    )
    supplied_total = _optional_number(
        gap_data.get("total_score", gap_data.get("current_score")), "total_score"
    )
    if supplied_total is not None and not math.isclose(
        supplied_total, calculated_total, abs_tol=0.2
    ):
        warnings.append(
            f"전달된 총점 {supplied_total:g} 대신 평가항목 합계 {calculated_total:g}을 사용했습니다."
        )
    target = _optional_number(gap_data.get("target_score"), "target_score")
    if target is None:
        target = calculated_target or 100.0
    if target < calculated_total:
        target = calculated_total

    idea = gap_data.get("idea")
    if not isinstance(idea, Mapping):
        idea = {
            "idea_id": gap_data.get("idea_id"),
            "name": gap_data.get("idea_name"),
        }

    return {
        "diagnosis_mode": diagnosis_mode,
        "score_contract": score_contract,
        "scores": scores,
        "category_details": details,
        "missing": missing,
        "score_summary": {
            "current_score": _tidy(calculated_total),
            "target_score": _tidy(target),
            "score_gap": _tidy(max(0.0, target - calculated_total)),
            "scored_maximum": _tidy(calculated_target),
            "fixed_organization_maximum": 55,
            "llm_added_maximum": (
                45 if diagnosis_mode == DIAGNOSIS_MODE_IDEA_FIT else 0
            ),
        },
        "idea": _json_safe_mapping(idea),
        "organization_profile": gap_data.get("organization_profile") or {},
        "score_reasons": gap_data.get("score_reasons") or [],
        "warnings": warnings,
        "context": {
            "diagnosis_mode": diagnosis_mode,
            "score_contract": score_contract,
            "organization_id": gap_data.get("organization_id"),
            "idea_id": idea.get("idea_id") or gap_data.get("idea_id"),
            "idea_name": idea.get("name") or gap_data.get("idea_name"),
        },
    }


def _normalize_category_details(
    raw_details: Any,
    scores: Mapping[str, Mapping[str, Any]],
    diagnosis_mode: str,
) -> OrderedDict[str, OrderedDict[str, dict[str, Any]]]:
    raw_details = raw_details if isinstance(raw_details, Mapping) else {}
    result: OrderedDict[str, OrderedDict[str, dict[str, Any]]] = OrderedDict()
    for category, definition in CATEGORY_DEFINITIONS.items():
        raw_category = raw_details.get(category)
        raw_category = raw_category if isinstance(raw_category, Mapping) else {}
        normalized_subitems: OrderedDict[str, dict[str, Any]] = OrderedDict()
        for subitem, (max_score, display_name) in definition["subitems"].items():
            raw = raw_category.get(subitem)
            score = None
            evidence = ""
            if isinstance(raw, Mapping):
                score = _optional_number(
                    raw.get("score"), f"category_details.{category}.{subitem}.score"
                )
                evidence = str(raw.get("evidence") or "").strip()
            elif raw is not None:
                score = _optional_number(
                    raw, f"category_details.{category}.{subitem}"
                )
            if score is not None and not 0 <= score <= float(max_score):
                score = None
            normalized_subitems[subitem] = {
                "display_name": display_name,
                "score": _tidy(score),
                "max_score": max_score,
                "data_status": (
                    "not_provided"
                    if score is None
                    else "llm_estimated"
                    if (
                        diagnosis_mode == DIAGNOSIS_MODE_IDEA_FIT
                        and category in LLM_ADDED_CATEGORIES
                    )
                    else "evidence_based"
                ),
                "score_origin": _score_origin(diagnosis_mode, category),
                "evidence": evidence,
            }
        result[category] = normalized_subitems
    return result


def _normalize_diagnosis_mode(raw_mode: Any) -> str:
    aliases = {
        "recommendation": DIAGNOSIS_MODE_RECOMMENDATION,
        "capability": DIAGNOSIS_MODE_RECOMMENDATION,
        "capability_recommendation": DIAGNOSIS_MODE_RECOMMENDATION,
        "idea": DIAGNOSIS_MODE_IDEA_FIT,
        "idea_fit": DIAGNOSIS_MODE_IDEA_FIT,
    }
    mode = aliases.get(str(raw_mode or "capability_recommendation").strip().lower())
    if mode is None:
        raise F5InputError(
            "diagnosis_mode는 capability_recommendation 또는 idea_fit이어야 합니다."
        )
    return mode


def _score_origin(diagnosis_mode: str, category: str) -> str:
    if category in ORGANIZATION_CATEGORIES:
        return "organization_score_engine"
    if diagnosis_mode == DIAGNOSIS_MODE_IDEA_FIT:
        return "llm_estimate_plus_score_engine"
    return "database_score_engine"


def _normalize_missing(raw_missing: Any) -> list[dict[str, str]]:
    if raw_missing is None:
        return []
    items: list[Any] = []
    if isinstance(raw_missing, Mapping):
        for level in ("core", "supporting"):
            for value in raw_missing.get(level, []) or []:
                items.append({"capability": value, "level": level})
    elif isinstance(raw_missing, Sequence) and not isinstance(
        raw_missing, (str, bytes, bytearray)
    ):
        items = list(raw_missing)
    else:
        raise F5InputError("missing은 리스트 또는 core/supporting 딕셔너리여야 합니다.")

    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in items:
        if isinstance(item, Mapping):
            name = str(item.get("capability") or item.get("name") or "").strip()
            level = str(item.get("level") or "unspecified").strip()
        else:
            name = str(item).strip()
            level = "unspecified"
        if not name or name in seen:
            continue
        seen.add(name)
        result.append({"capability": name, "level": level})
    return result[:12]


def _derive_score_based_gaps(
    scores: Mapping[str, Mapping[str, Any]],
    details: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> list[dict[str, str]]:
    """명시적 missing이 없을 때 비만점 세부항목에서 보완과제를 만든다."""

    candidates: list[tuple[float, float, str]] = []
    for category, definition in CATEGORY_DEFINITIONS.items():
        category_detail_found = False
        for subitem, (_maximum, display_name) in definition["subitems"].items():
            detail = details.get(category, {}).get(subitem, {})
            score = detail.get("score")
            maximum = detail.get("max_score")
            if score is None or maximum in (None, 0) or _is_full(score, maximum):
                continue
            category_detail_found = True
            score_gap = float(maximum) - float(score)
            gap_ratio = score_gap / float(maximum)
            candidates.append((gap_ratio, score_gap, display_name))

        category_score = scores.get(category, {}).get("score")
        category_maximum = scores.get(category, {}).get("max_score")
        if (
            not category_detail_found
            and category_score is not None
            and category_maximum not in (None, 0)
            and not _is_full(category_score, category_maximum)
        ):
            score_gap = float(category_maximum) - float(category_score)
            candidates.append(
                (
                    score_gap / float(category_maximum),
                    score_gap,
                    f"{category} 보완",
                )
            )

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for _ratio, _gap, capability in candidates:
        if capability in seen:
            continue
        seen.add(capability)
        result.append({"capability": capability, "level": "score_based"})
        if len(result) == 3:
            break
    return result


def _build_sources(
    reference_sources: Sequence[Mapping[str, Any]],
    organization_profile: Mapping[str, Any],
    score_reasons: Sequence[Any],
) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    seen: set[str] = set()

    def add(source_id: str, label: str, content: str, source_type: str) -> None:
        source_id = source_id.strip()
        content = " ".join(content.split())
        if not source_id or not content or source_id in seen:
            return
        seen.add(source_id)
        result.append(
            {
                "source_id": source_id,
                "label": label.strip() or source_id,
                "content": content[:4_000],
                "source_type": source_type,
            }
        )

    for index, source in enumerate(reference_sources):
        if not isinstance(source, Mapping):
            continue
        source_id = str(source.get("source_id") or f"idea_source_{index + 1}")
        add(
            source_id,
            str(source.get("idea_name") or source.get("label") or "아이디어 데이터"),
            str(source.get("content") or ""),
            str(source.get("source_type") or "idea_data"),
        )

    if isinstance(organization_profile, Mapping):
        capabilities = []
        for item in organization_profile.get("capabilities", []) or []:
            if not isinstance(item, Mapping):
                continue
            name = str(item.get("name") or item.get("capability_id") or "").strip()
            if name:
                capabilities.append(f"{name}(보유수준 {item.get('level', '미확인')})")
        if capabilities:
            add(
                "org_capabilities",
                "조직 기술역량 프로필",
                "; ".join(capabilities[:30]),
                "organization_data",
            )

        staff_roles = []
        for item in organization_profile.get("staff", []) or []:
            if not isinstance(item, Mapping):
                continue
            role = item.get("담당") or item.get("role") or item.get("직무")
            career = item.get("경력") or item.get("experience")
            text = " / ".join(str(v).strip() for v in (role, career) if v)
            if text:
                staff_roles.append(text)
        if staff_roles:
            add(
                "org_staff_roles",
                "조직 인력역량",
                "; ".join(staff_roles[:25]),
                "organization_data",
            )

        patents = []
        for evidence in (organization_profile.get("evidence") or {}).values():
            if not isinstance(evidence, Mapping) or evidence.get("type") != "patent":
                continue
            title = str(evidence.get("title") or "").strip()
            classification = str(evidence.get("classification") or "").strip()
            if title:
                patents.append(f"[{classification or '미분류'}] {title}")
        if patents:
            add(
                "org_patents",
                "조직 특허 포트폴리오",
                "; ".join(patents[:35]),
                "organization_data",
            )

        signals = organization_profile.get("signals")
        if isinstance(signals, Mapping) and signals:
            add(
                "org_signals",
                "조직 역량 갭 신호",
                json.dumps(signals, ensure_ascii=False, default=str),
                "organization_data",
            )

    for index, reason in enumerate(score_reasons):
        if isinstance(reason, Mapping):
            category = str(reason.get("category") or f"평가항목 {index + 1}")
            content = str(reason.get("reason") or reason.get("content") or "")
        elif isinstance(reason, Sequence) and not isinstance(reason, str) and len(reason) >= 2:
            category, content = str(reason[0]), str(reason[1])
        else:
            continue
        add(
            f"score_reason_{index + 1}",
            f"{category} 점수 산정 근거",
            content,
            "score_input",
        )

    return result[:50]


def _build_slots(normalized: Mapping[str, Any]) -> dict[str, Any]:
    """LLM이 실제로 작성해야 하는 비만점 슬롯만 만든다."""

    categories: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for category_index, (category, definition) in enumerate(
        CATEGORY_DEFINITIONS.items(), start=1
    ):
        score_info = normalized["scores"][category]
        score = score_info["score"]
        maximum = score_info["max_score"]
        if score is None or math.isclose(float(score), float(maximum), abs_tol=0.01):
            continue

        subitems: OrderedDict[str, dict[str, str]] = OrderedDict()
        for subitem_index, (subitem, values) in enumerate(
            definition["subitems"].items(), start=1
        ):
            detail = normalized["category_details"][category][subitem]
            sub_score = detail["score"]
            sub_maximum = detail["max_score"]
            if sub_score is None or math.isclose(
                float(sub_score), float(sub_maximum), abs_tol=0.01
            ):
                continue
            subitems[f"s{category_index:02d}_{subitem_index:02d}"] = {
                "key": subitem,
                "display_name": values[1],
            }
        categories[f"c{category_index:02d}"] = {
            "category": category,
            "subitems": subitems,
        }

    gaps = OrderedDict(
        (
            f"g{index:02d}",
            {"capability": item["capability"], "level": item["level"]},
        )
        for index, item in enumerate(normalized["missing"], start=1)
    )
    return {"categories": categories, "gaps": gaps}


def _text_node_schema(properties: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": dict(properties),
        "required": list(properties),
    }


def _build_output_schema(slots: Mapping[str, Any]) -> dict[str, Any]:
    category_properties: OrderedDict[str, Any] = OrderedDict()
    for category_slot, meta in slots["categories"].items():
        subitem_properties = OrderedDict(
            (
                subitem_slot,
                _text_node_schema(
                    {
                        "assessment": {
                            "type": "string",
                            "description": "조직 보유 수준과 아이디어 요구 수준의 비교 판단. 한국어 200~350자.",
                        },
                        "assessment_brief": {
                            "type": "string",
                            "description": "assessment를 대시보드용으로 줄인 한국어 150~180자.",
                        },
                        "improvement": {
                            "type": "string",
                            "description": (
                                "한국어 800~1500자, 최우선 항목은 1200~2000자. 현재 기반, "
                                "구체적 공백, 확보 대상, 실행 순서, 담당 역할, 검증 산출물과 "
                                "완료 기준을 설명한다."
                            ),
                        },
                        "improvement_brief": {
                            "type": "string",
                            "description": (
                                "improvement를 대시보드용으로 줄인 한국어 260~300자. "
                                "현재 기반·핵심 공백·실행 방향·완료 기준을 남긴다."
                            ),
                        },
                    }
                ),
            )
            for subitem_slot in meta["subitems"]
        )
        category_properties[category_slot] = _text_node_schema(
            {
                "headline": {
                    "type": "string",
                    "description": "카드에 표시할 핵심 결론 한 문장. 한국어 60~120자.",
                },
                "summary": {
                    "type": "string",
                    "description": "평가항목 종합분석. 원인·영향·핵심 보완방향을 한국어 250~450자로 작성.",
                },
                "summary_brief": {
                    "type": "string",
                    "description": "summary를 대시보드용으로 줄인 한국어 170~200자.",
                },
                "subitems": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": subitem_properties,
                    "required": list(subitem_properties),
                },
            }
        )

    gap_properties = OrderedDict(
        (
            slot,
            _text_node_schema(
                {
                    "headline": {
                        "type": "string",
                        "description": "부족 원인과 우선 확보 방향을 압축한 한국어 70~140자 한 문장.",
                    },
                    "cause": {
                        "type": "string",
                        "description": "구체적인 부족 원인. 한국어 250~400자.",
                    },
                    "cause_brief": {
                        "type": "string",
                        "description": "cause를 대시보드용으로 줄인 한국어 250~300자.",
                    },
                    "impact": {
                        "type": "string",
                        "description": "실행·고객·비용·규제·사업 측면의 영향. 한국어 250~400자.",
                    },
                    "impact_brief": {
                        "type": "string",
                        "description": "impact를 대시보드용으로 줄인 한국어 250~300자.",
                    },
                    "priority_strategy": {
                        "type": "string",
                        "enum": list(STRATEGY_TYPES),
                        "description": "가장 먼저 적용할 확보 방향 하나. build, buy, partner, hire 중 선택.",
                    },
                    "priority_rationale": {
                        "type": "string",
                        "description": "해당 확보 방향을 우선하는 진단상 이유. 세부 실행안 없이 한국어 150~300자.",
                    },
                    "priority_rationale_brief": {
                        "type": "string",
                        "description": "priority_rationale을 대시보드용으로 줄인 한국어 170~200자.",
                    },
                }
            ),
        )
        for slot in slots["gaps"]
    )

    strategy_node = _text_node_schema(
        {
            "headline": {
                "type": "string",
                "description": "카드에 표시할 전략별 핵심 실행안. 한국어 80~160자 한 문장.",
            },
            "applicability": {
                "type": "string",
                "enum": ["recommended", "conditional", "not_applicable"],
            },
            "action": {
                "type": "string",
                "description": "아이디어에 특화된 실행안. 해당 부족역량의 네 전략 전체가 2000~2800자가 되도록 작성.",
            },
            "action_brief": {
                "type": "string",
                "description": "action을 대시보드용으로 줄인 한국어 250~300자.",
            },
            "rationale": {
                "type": "string",
                "description": "적용 여부와 선택 조건을 설명하는 판단 근거. 해당 부족역량의 네 전략 전체가 2000~2800자가 되도록 작성.",
            },
            "rationale_brief": {
                "type": "string",
                "description": "rationale을 대시보드용으로 줄인 한국어 210~250자.",
            },
        }
    )
    strategy_properties = OrderedDict(
        (
            gap_slot,
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {key: strategy_node for key in STRATEGY_TYPES},
                "required": list(STRATEGY_TYPES),
            },
        )
        for gap_slot in slots["gaps"]
    )

    roadmap_phase = _text_node_schema(
        {
            "headline": {
                "type": "string",
                "description": "단계 카드에 표시할 핵심 목표와 산출물. 한국어 80~160자 한 문장.",
            },
            "objective": {
                "type": "string",
                "description": "해당 단계가 끝났을 때 달성할 구체적 상태. 단계 전체 1200~1800자 중 한국어 200~300자.",
            },
            "objective_brief": {
                "type": "string",
                "description": "objective를 대시보드용으로 줄인 한국어 170~200자.",
            },
            "actions": {
                "type": "array",
                "items": {
                    "type": "string",
                    "description": "담당·대상·방법·산출물이 드러나는 실행 항목. 3개, 항목당 한국어 150~250자.",
                },
            },
            "actions_brief": {
                "type": "array",
                "description": "actions를 대시보드용으로 줄인 목록. actions와 같은 개수·같은 순서.",
                "items": {
                    "type": "string",
                    "description": "같은 순서의 action을 줄인 한국어 55~70자.",
                },
            },
            "completion_criteria": {
                "type": "array",
                "items": {
                    "type": "string",
                    "description": "완료 여부를 판단할 수 있는 검증 기준. 2~3개, 항목당 한국어 100~160자.",
                },
            },
            "completion_criteria_brief": {
                "type": "array",
                "description": "completion_criteria를 줄인 목록. 같은 개수·같은 순서.",
                "items": {
                    "type": "string",
                    "description": "같은 순서의 기준을 줄인 한국어 70~90자.",
                },
            },
        }
    )

    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "executive_headline": {
                "type": "string",
                "description": "메인 화면에 표시할 최종 판단. 한국어 100~180자, 1~2문장.",
            },
            "executive_summary": {
                "type": "string",
                "description": "핵심 강점·최우선 갭·우선 전략을 담은 한국어 500~800자 요약.",
            },
            "categories": {
                "type": "object",
                "additionalProperties": False,
                "properties": category_properties,
                "required": list(category_properties),
            },
            "gaps": {
                "type": "object",
                "additionalProperties": False,
                "properties": gap_properties,
                "required": list(gap_properties),
            },
            "strategies": {
                "type": "object",
                "additionalProperties": False,
                "properties": strategy_properties,
                "required": list(strategy_properties),
            },
            "roadmap": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    phase: roadmap_phase for phase in ROADMAP_PHASES
                },
                "required": list(ROADMAP_PHASES),
            },
            "unverified_items": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": [
            "executive_headline",
            "executive_summary",
            "categories",
            "gaps",
            "strategies",
            "roadmap",
            "unverified_items",
        ],
    }


def _build_system_prompt(normalized: Mapping[str, Any]) -> str:
    if normalized["diagnosis_mode"] == DIAGNOSIS_MODE_IDEA_FIT:
        mode_rules = """
[현재 진단 모드: 아이디어 적합도 판단형]
- 조직역량적합도·역량전이가능성·부족역량수준·실행가능성의 55점은 조직 점수엔진 결과입니다.
- 시장성·경쟁강도·진입장벽·사업성의 45점은 앞 단계 LLM이 아이디어의 시장규모와
  진입장벽을 추정하고 공통 F2-5 산식이 계산한 값입니다. DB 실측값으로 표현하지 마세요.
- 이 45점은 아이디어 적합도 판단을 위한 추정 평가이므로, 분석에는 반드시 실제로
  확인해야 할 시장자료·경쟁사례·규제조건·고객검증 항목을 포함하세요.
- F5에서 45점을 다시 계산하거나 임의로 바꾸지 말고, 읽기 전용 추정 점수로 해석하세요.
"""
    else:
        mode_rules = """
[현재 진단 모드: 역량 기반 추천형]
- 8개 항목 100점은 조직 데이터와 신사업 DB 데이터를 공통 점수엔진으로 계산한 값입니다.
- 시장성·경쟁강도·진입장벽·사업성은 제공된 DB 원문과 점수 산정 근거를 우선 사용하세요.
"""

    return ("""
당신은 기업의 신사업 진입 가능성을 진단하고 실행 가능한 보완안을 쓰는 F5 전략
분석가입니다. 점수 계산은 이미 끝났습니다. 당신의 일은 입력된 조직 자료, 아이디어,
점수 산정 근거와 참고 자료를 연결해 비만점 평가항목의 보완점을 구체적으로 설명하는
것뿐입니다.

{mode_rules}

[반드시 지킬 규칙]
1. c01, s01_01, g01 같은 슬롯 키를 삭제·추가·변경하지 마세요.
2. 입력의 점수·배점·평가항목명·세부항목명은 읽기 전용이며 다시 계산하지 마세요.
3. 입력에는 보완이 필요한 비만점 항목만 들어옵니다. 만점 항목은 파이썬이 별도로
   '추가 보완 불필요'라고 표시하므로 추측해서 만들지 마세요.
4. source_id, 출처 종류, data_status, 점수 출처를 출력하지 마세요. 파이썬이 붙입니다.
5. DB 자료가 부족해도 일반적인 전문지식을 바탕으로 합리적인 보완안을 제안할 수
   있습니다. 다만 입력에 없는 기업·제품·특허·인증을 이미 보유하거나 적용 중인
   사실처럼 단정하지 마세요.
6. 입력에 없는 기간·인원·데이터 수량·성능 목표는 '권고 범위' 또는 '예시'라고
   분명히 표현하세요. 보완 시 점수가 몇 점 오른다고 예측하지 마세요.

[사용자 표시 문체]
- 전문 컨설턴트가 비전문가인 사용자에게 분석 결과와 다음 행동을 차분하게 설명하는
  자연스러운 한국어 존댓말을 사용하세요.
- headline을 제외한 모든 본문 문장은 하십시오체로 작성하되, 관공서 문서처럼
  딱딱한 표현은 피하세요. '~합니다', '~됩니다', '~필요합니다', '~권장합니다',
  '~하는 것이 좋습니다', '~할 필요가 있습니다'를 문맥에 맞게 자연스럽게 섞으세요.
- 본문에서 '~한다', '~이다', '~된다', '~해야 한다', '~것이다' 같은 한다체를 사용하지
  마세요. 제목·배지·짧은 소제목만 명사형으로 작성할 수 있습니다.
- 사용자를 훈계하거나 단정적으로 명령하지 마세요. 먼저 현재 기반과 판단 이유를
  설명한 뒤, 구체적인 보완 행동을 권고하는 순서로 작성하세요.
- 같은 종결어미를 기계적으로 반복하지 말고 문장 길이와 연결어를 자연스럽게
  조절하세요. 전문용어는 해당 아이디어에서 무엇을 위해 필요한지도 함께 설명하세요.
- 권장 문체 예시:
  '현재 조직은 의료영상 진단과 데이터 품질 관리에서 높은 수준의 기술역량을
  보유하고 있습니다. 다만 현재 성과는 개별 진단모델 개발에 집중되어 있어 병원 간
  데이터 표준화와 사용권 관리까지 바로 확장하기에는 추가적인 준비가 필요합니다.
  초기에는 승인된 샘플 데이터로 전체 처리과정을 시험하는 것이 좋습니다. 이 과정에서
  데이터 사전과 기관별 매핑표, 품질검사 결과를 함께 확보할 필요가 있습니다.'

[요구하는 구체성]
- executive_headline은 사용자가 처음 읽을 최종 판단입니다. 아이디어의 핵심 강점,
  최우선 보완과제와 우선 확보 방향을 한국어 100~180자, 1~2문장으로 압축하세요.
- executive_summary는 한국어 500~800자로 핵심 강점, 최우선 갭 2~3개, 우선 적용할
  전략과 실행 순서를 요약하세요.
- category headline은 점수 상태와 가장 중요한 보완방향을 한국어 60~120자 한 문장으로
  작성하고, gap headline은 구체적인 공백과 priority_strategy를 한국어 70~140자로
  압축하세요.
- category summary는 원인 → 사업상 영향 → 가장 중요한 보완 방향 순서로 한국어
  250~450자, 3~5문장으로 작성하세요.
- 각 subitem의 assessment는 조직이 실제로 보유한 것과 아이디어가 요구하는 것을
  비교해 한국어 200~350자, 2~3문장으로 판단하세요.
- 각 subitem의 improvement는 한국어 800~1500자로 작성하세요. 우선순위가 높고
  기술·데이터·특허·인력 등 여러 공백이 얽힌 항목은 1200~2000자를 사용하세요.
  분량을 채우기 위한 반복은 금지합니다.
- improvement에는 최소 4문장으로 다음 네 요소를 포함하세요.
  ① 현재 확보된 기반, ② 부족하거나 확인되지 않은 구체 영역, ③ 조사·개발·도입할
  기술·데이터·특허·인력·프로세스의 구체 명칭, ④ 실행 방법과 확인할 산출물.
- '강화하세요', '검토하세요', '전문가와 협력하세요' 한 문장으로 끝내지 마세요.
  무엇을 대상으로 누가 어떤 자료나 시스템을 만들고 무엇으로 완료를 확인할지 쓰세요.
- 특허 항목은 아이디어의 핵심 기능을 나누고, 추가 조사할 특허 분야·기능 방향,
  선행기술/청구항 비교, 신규 출원·공동개발·실시권 확보 중 적합한 경로를 설명하세요.
- 기술·데이터 항목은 필요한 데이터 형식, 수집 환경, 전처리·모델·연동 기능,
  대표 오류 유형과 검증 산출물을 설명하세요.
- 인력 항목은 필요한 직무와 그 담당자가 맡을 설계·검증·운영 업무를 설명하세요.
- 시장·경쟁·사업 항목은 고객군, 구매 의사결정자, 대체수단, 가격·판매채널 가설과
  고객 검증 자료를 설명하세요.
- 진입장벽 항목은 데이터 권리, 규제·인증 가능성, 계약, 보안·기록관리와 운영 통제를
  아이디어에 맞게 설명하세요.
- 평가항목마다 다른 원인과 행동을 쓰고 같은 문장을 반복하지 마세요.

[F5-1 · 부족 역량 진단 작성 기준]
- 부족 역량 소제목 하나당 cause·impact·priority_rationale을 합쳐 한국어
  650~1100자로 작성하세요. cause 250~400자, impact 250~400자,
  priority_rationale 150~300자를 권장합니다.
- cause에는 단순히 '역량이 부족하다'고 쓰지 말고, 현재 보유 기술·특허·인력·데이터와
  아이디어가 요구하는 기능·운영조건 사이의 구체적인 차이를 설명하세요.
- impact에는 개발 지연 같은 일반론만 쓰지 말고, 고객 도입, 연동, 품질, 규제·계약,
  비용, 매출 검증 중 이 아이디어에 실제로 영향을 주는 경로를 연결하세요.
- priority_strategy는 Build·Buy·Partner·Hire 중 가장 먼저 적용할 방향 하나만 선택하세요.
  F5-2에서 not_applicable로 판단할 전략을 선택하면 안 됩니다.
- priority_rationale에는 왜 그 방향이 현재 격차를 가장 빠르고 현실적으로 줄이는지만
  설명하세요. 담당자·일정·산출물·세부 실행방법은 F5-2와 중복되므로 쓰지 마세요.

[F5-2 · Build·Buy·Partner·Hire 작성 기준]
- 부족 역량 소제목 하나 아래의 Build·Buy·Partner·Hire 네 전략 전체를 합쳐 한국어
  2000~2800자로 작성하세요. recommended 전략은 카드당 500~700자, conditional은
  350~500자, not_applicable은 150~300자를 권장합니다. 복합적인 전략은 충분히
  자세히 쓰되 반복 문장으로 분량을 채우지 마세요.
- 각 부족 역량에는 recommended 또는 conditional 전략이 최소 하나 있어야 하며,
  F5-1의 priority_strategy와 같은 전략은 not_applicable로 표시하면 안 됩니다.
- 각 전략 headline에는 조사·강화 같은 추상어 대신 도입·개발·제휴·채용할 구체 대상을
  넣어 한국어 80~160자 한 문장으로 작성하세요. action과 rationale의 상세내용을 그대로
  반복하지 마세요.
- Build는 개발할 모듈·데이터·프로세스, 담당 역할, 시험 환경과 완료 산출물을 쓰세요.
- Buy는 도입할 기술·서비스·데이터·소프트웨어·라이선스의 유형, 공급사 비교 기준,
  PoC 항목, 데이터·지식재산·유지보수·종료 조건을 쓰세요. 외부 인력 채용을 Buy로
  표현하지 마세요.
- Partner는 필요한 파트너 유형, 각자의 역할, 제공 데이터·시설, 결과물과 지식재산의
  귀속, 성공·중단 조건을 쓰세요.
- Hire는 필요한 직무명, 맡길 설계·검증·운영 업무, 요구 경력이나 포트폴리오,
  입사 또는 배치 후 첫 산출물을 쓰세요.
- not_applicable인 전략도 상투적인 문구로 끝내지 말고, 이 아이디어에서 부적합한 이유와
  어떤 조건이 바뀌면 다시 검토할 수 있는지를 설명하세요.

[F5-3 · 실행 로드맵 작성 기준]
- 단기·중기·장기 각 단계의 objective·actions·completion_criteria를 합쳐 한국어
  1200~1800자로 작성하세요. 기간 숫자는 입력 근거가 없으면 권고 범위임을 밝히세요.
- 각 단계 headline은 핵심 목표와 다음 단계로 넘길 산출물을 한국어 80~160자 한 문장으로
  작성하세요.
- 각 단계는 앞 단계의 산출물을 다음 단계가 실제로 이어받도록 연결하세요.
- actions는 3개를 권장하며, 각 항목에 담당 역할, 대상, 수행 방법과 산출물을
  포함하세요. completion_criteria는 2~3개를 권장하며 문서 제출 같은 형식적 기준만
  쓰지 말고 품질·고객·운영·규제 검증 결과로 통과와 중단을 판단할 수 있게 하세요.
- '조사한다 → 파일럿한다 → 출시한다'라는 일반적인 문장을 반복하지 말고, 입력된
  아이디어의 기술명·데이터 형식·고객군·운영환경을 사용해 단계별 범위를 구분하세요.

[대시보드용 짧은 버전(*_brief) 작성 기준]
- 이름이 '_brief'로 끝나는 필드는 바로 앞의 원문 필드를 화면 카드에 넣기 위해
  줄인 것입니다. 원문 필드는 인쇄용 보고서에 그대로 쓰이므로 길이를 줄이지 마세요.
- 각 _brief는 해당 필드 설명에 적힌 글자수 범위를 지키고, 그 범위의 90~100%를
  채우세요. 너무 짧게 줄이면 화면이 허전해집니다.
- 원문에 없는 사실·숫자를 새로 만들지 말고, 원문의 결론과 권고 방향을 유지하세요.
- 숫자와 고유명사(기술명·도구명·기간 등)는 최대한 살리세요.
- 마지막 문장까지 반드시 완결하고 '~습니다/~합니다'로 끝내세요. 말줄임표는 쓰지
  마세요. 정해진 글자수 안에서 끝낼 수 있도록 문장 수를 미리 계획해 쓰세요.
- actions_brief와 completion_criteria_brief는 원문 목록과 개수·순서가 같아야 합니다.

[품질 기준 예시 — 문장 구조와 구체성만 따르고 의료 분야 표현을 다른 아이디어에
그대로 복사하지 마세요]
"현재 보유 특허는 의료영상 분석과 병변 탐지 기술에 집중되어 있지만, 해당 아이디어의
핵심 기능인 의료기관 간 데이터 표준화와 임상 시스템 연동에 대한 권리 범위는
확인되지 않습니다. 추가로 DICOM 메타데이터 정규화, PACS·EMR 연동, 의료영상
비식별화, 이기종 장비 간 화질 편차 보정 분야의 특허를 조사하는 것이 좋습니다.
자체 출원이 어렵다면 병원 데이터 연계 및 DICOM 변환 기술을 보유한 기업의 특허를
대상으로 실시권 확보 가능성을 검토해야 합니다."

[Build·Buy·Partner·Hire]
- Build: 내부 기술·데이터·프로세스·운영역량을 개발하고 축적합니다.
- Buy: 기업 인수가 아니라 외부 기술·서비스·데이터·소프트웨어·라이선스를 도입합니다.
- Partner: 병원·대학·연구기관·전문기관·공급사와 역할을 나누어 공동 검증합니다.
- Hire: 필요한 전문 역할을 정의하고 채용 또는 전담 배치합니다.
- 네 전략을 억지로 모두 권장하지 마세요. 부적합하면 not_applicable로 쓰되 action과
  rationale에 왜 현재 적용하지 않는지 한 문장씩 작성하세요.

[로드맵]
- 단기: 확인할 요구사항과 작은 검증 과제, 담당자, 산출물을 제시하세요.
- 중기: 파일럿 범위, 외부 협력 또는 도입 검증, 성공·중단 판단자료를 제시하세요.
- 장기: 검증된 방식의 운영 정착, 확장 조건, 재평가 자료를 제시하세요.

지정된 JSON Schema에 맞는 JSON 객체만 출력하세요.
""".format(mode_rules=mode_rules.strip()).strip())


def _build_user_content(
    normalized: Mapping[str, Any],
    sources: Sequence[Mapping[str, str]],
    slots: Mapping[str, Any],
) -> str:
    category_inputs: OrderedDict[str, Any] = OrderedDict()
    for category_slot, meta in slots["categories"].items():
        category = meta["category"]
        score_info = normalized["scores"][category]
        subitems: OrderedDict[str, Any] = OrderedDict()
        for subitem_slot, submeta in meta["subitems"].items():
            detail = normalized["category_details"][category][submeta["key"]]
            subitems[subitem_slot] = {
                "name": submeta["display_name"],
                "score": detail["score"],
                "max_score": detail["max_score"],
                "evidence": detail["evidence"],
            }
        category_inputs[category_slot] = {
            "name": category,
            "meaning": CATEGORY_DEFINITIONS[category]["meaning"],
            "score": score_info["score"],
            "max_score": score_info["max_score"],
            "subitems": subitems,
        }

    payload = {
        "diagnosis_mode": normalized["diagnosis_mode"],
        "score_contract": normalized["score_contract"],
        "idea": normalized["idea"],
        "score_summary_read_only": normalized["score_summary"],
        "categories_read_only": category_inputs,
        "gaps_read_only": slots["gaps"],
        "reference_information": [
            {
                "label": source["label"],
                "source_type": source["source_type"],
                "content": source["content"],
            }
            for source in sources
        ],
        "instruction": (
            "각 슬롯에 해당 아이디어 전용 분석을 작성하세요. 품질 기준 예시와 같은 "
            "구체성을 유지하되, 입력에 있는 실제 아이디어·조직 역량·데이터·특허·인력에 "
            "맞춰 내용을 바꾸세요."
        ),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def _call_openai(
    system_prompt: str,
    user_content: str,
    model: str,
    output_schema: Mapping[str, Any],
) -> str:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # Streamlit Cloud 등 배포 환경에서는 secrets.toml에만 키가 있고
        # os.environ에는 없을 수 있으므로 st.secrets도 확인한다.
        try:
            import streamlit as st

            api_key = st.secrets.get("OPENAI_API_KEY")
        except Exception:
            api_key = None
    if not api_key:
        raise F5ConfigurationError(
            "OPENAI_API_KEY가 없습니다. app/.env 또는 Streamlit secrets를 확인하세요."
        )
    try:
        import openai
    except ImportError as exc:
        raise F5ConfigurationError(
            "openai 패키지가 없습니다. requirements.txt를 설치하세요."
        ) from exc

    client = openai.OpenAI(api_key=api_key, timeout=600.0, max_retries=1)
    request_options: dict[str, Any] = {
        "model": model,
        "instructions": system_prompt,
        "input": user_content,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "store": False,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "f5_gap_report_rebuild",
                "schema": output_schema,
                "strict": True,
            }
        },
    }
    if model.startswith("gpt-5.6"):
        reasoning_effort = os.getenv(
            "F5_REASONING_EFFORT", DEFAULT_REASONING_EFFORT
        ).strip().lower()
        if reasoning_effort not in {"none", "low", "medium", "high", "xhigh", "max"}:
            reasoning_effort = DEFAULT_REASONING_EFFORT
        request_options["reasoning"] = {"effort": reasoning_effort}
        text_verbosity = os.getenv(
            "F5_TEXT_VERBOSITY", DEFAULT_TEXT_VERBOSITY
        ).strip().lower()
        if text_verbosity not in {"low", "medium", "high"}:
            text_verbosity = DEFAULT_TEXT_VERBOSITY
        request_options["text"]["verbosity"] = text_verbosity
    try:
        response = client.responses.create(**request_options)
    except openai.AuthenticationError as exc:
        raise F5ConfigurationError("OpenAI API 키 인증에 실패했습니다.") from exc
    except openai.RateLimitError as exc:
        raise F5ProviderError("OpenAI 사용 한도 또는 결제 잔액을 확인하세요.") from exc
    except (openai.APIConnectionError, openai.APITimeoutError) as exc:
        raise F5ProviderError("OpenAI 연결이 지연되거나 시간 제한을 초과했습니다.") from exc
    except openai.APIStatusError as exc:
        raise F5ProviderError(
            f"OpenAI API 호출이 실패했습니다(상태 코드 {getattr(exc, 'status_code', '?')})."
        ) from exc

    if getattr(response, "status", None) == "incomplete":
        reason = getattr(getattr(response, "incomplete_details", None), "reason", None)
        raise F5ResponseError(
            "LLM 응답이 완료되지 않았습니다. "
            + ("출력 길이 제한에 도달했습니다." if reason == "max_output_tokens" else str(reason))
        )
    output_text = getattr(response, "output_text", None)
    if not isinstance(output_text, str) or not output_text.strip():
        raise F5ResponseError("OpenAI 응답에 분석 JSON이 없습니다.")
    return output_text


def _parse_json_response(raw_response: Any) -> dict[str, Any]:
    if isinstance(raw_response, Mapping):
        return copy.deepcopy(dict(raw_response))
    if not isinstance(raw_response, str) or not raw_response.strip():
        raise F5ResponseError("LLM 응답이 비어 있습니다.")
    text = raw_response.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise F5ResponseError(f"LLM 응답 JSON을 읽을 수 없습니다: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise F5ResponseError("LLM 응답 최상위 값은 JSON 객체여야 합니다.")
    return payload


def _assert_required_slots(payload: Mapping[str, Any], slots: Mapping[str, Any]) -> None:
    """구조와 슬롯 개수만 확인한다. 문장 내용·출처·수치는 검증하지 않는다."""

    required_top = {
        "executive_headline",
        "executive_summary",
        "categories",
        "gaps",
        "strategies",
        "roadmap",
        "unverified_items",
    }
    if not required_top.issubset(payload):
        raise F5ResponseError("LLM 구조화 응답에 필수 섹션이 없습니다.")
    if set(payload.get("categories", {})) != set(slots["categories"]):
        raise F5ResponseError("평가항목 분석 슬롯이 입력과 다릅니다.")
    if set(payload.get("gaps", {})) != set(slots["gaps"]):
        raise F5ResponseError("부족역량 분석 슬롯이 입력과 다릅니다.")
    if set(payload.get("strategies", {})) != set(slots["gaps"]):
        raise F5ResponseError("보완전략 슬롯이 입력과 다릅니다.")
    for category_slot, meta in slots["categories"].items():
        node = payload["categories"].get(category_slot, {})
        if set(node.get("subitems", {})) != set(meta["subitems"]):
            raise F5ResponseError(f"{category_slot}의 세부항목 슬롯이 입력과 다릅니다.")
    if set(payload.get("roadmap", {})) != set(ROADMAP_PHASES):
        raise F5ResponseError("실행 로드맵 단계가 입력과 다릅니다.")


def _is_full(score: Any, maximum: Any) -> bool:
    return score is not None and math.isclose(
        float(score), float(maximum), abs_tol=0.01
    )


def _priority(score: Any, maximum: Any) -> str:
    if score is None:
        return "not_scored"
    if _is_full(score, maximum):
        return "maintain"
    ratio = (float(maximum) - float(score)) / float(maximum)
    if ratio >= 0.5:
        return "high"
    if ratio >= 0.25:
        return "medium"
    return "low"


def _full_category_summary(score: Any, maximum: Any) -> str:
    return (
        f"현재 {score}/{maximum}점으로 만점입니다. 이 평가항목은 추가 보완이 "
        "필요하지 않으며, 현재 확보한 강점과 운영 수준을 유지하면 됩니다."
    )


def _source_evidence(
    normalized: Mapping[str, Any],
    sources: Sequence[Mapping[str, str]],
    *,
    category: str | None = None,
    subitem: str | None = None,
) -> dict[str, Any]:
    """LLM 판단이 아니라 입력 자료의 종류에 따라 출처 표시를 붙인다."""

    selected: list[Mapping[str, str]] = []

    def add(source: Mapping[str, str]) -> None:
        if source not in selected:
            selected.append(source)

    if category:
        for source in sources:
            if category in source.get("label", ""):
                add(source)

        if category in ORGANIZATION_CATEGORIES:
            preferred = ["org_capabilities"]
            if subitem and subitem.startswith("B"):
                preferred.insert(0, "org_patents")
            elif subitem and subitem.startswith("C"):
                preferred.insert(0, "org_staff_roles")
            elif subitem and subitem.startswith("F"):
                preferred.insert(0, "org_signals")
            elif subitem is None:
                preferred.extend(["org_patents", "org_staff_roles", "org_signals"])
            for source_id in preferred:
                for source in sources:
                    if source.get("source_id") == source_id:
                        add(source)
        else:
            for source in sources:
                if source.get("source_type") == "idea_data":
                    add(source)
    else:
        for source in sources:
            if source.get("source_type") in {
                "idea_data",
                "organization_data",
                "score_input",
            }:
                add(source)

    selected = selected[:4]
    source_ids = [source["source_id"] for source in selected]
    if source_ids:
        basis = "mixed"
    else:
        basis = "llm_recommendation"
    return {
        "basis": basis,
        "basis_label": _basis_label(basis),
        "source_ids": source_ids,
        "source_labels": [source["label"] for source in selected],
    }


def _assemble_report(
    normalized: Mapping[str, Any],
    payload: Mapping[str, Any],
    sources: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    slots = _build_slots(normalized)
    slot_by_category = {
        meta["category"]: (category_slot, meta)
        for category_slot, meta in slots["categories"].items()
    }
    categories: list[dict[str, Any]] = []
    for category, definition in CATEGORY_DEFINITIONS.items():
        score_info = normalized["scores"][category]
        score = score_info["score"]
        maximum = score_info["max_score"]
        slot_info = slot_by_category.get(category)
        llm_category = (
            payload["categories"][slot_info[0]] if slot_info else None
        )

        # 고정 문구로 채우는 분기(점수 없음·만점)는 이미 짧아서 줄일 게 없다.
        # LLM이 쓴 분기에서만 *_brief를 채우고, 나머지는 원문을 그대로 쓴다.
        summary_brief = ""
        if score is None:
            headline = "점수 산정 근거를 먼저 확인해야 하는 평가항목입니다."
            summary = (
                "이 평가항목은 앞 단계 점수 자료가 없어 보완 필요성을 확정할 수 "
                "없습니다. 점수 산정 근거를 먼저 확인해야 합니다."
            )
        elif _is_full(score, maximum):
            headline = "현재 만점 수준으로, 확보된 강점을 유지하면 됩니다."
            summary = _full_category_summary(score, maximum)
        else:
            headline = llm_category["headline"]
            summary = llm_category["summary"]
            summary_brief = _limit_text(
                llm_category.get("summary_brief"), CATEGORY_SUMMARY_BRIEF_MAX_CHARS
            )

        subslot_by_key = {
            submeta["key"]: subitem_slot
            for subitem_slot, submeta in (slot_info[1]["subitems"].items() if slot_info else [])
        }
        subitems: list[dict[str, Any]] = []
        for subitem, (_sub_maximum, display_name) in definition["subitems"].items():
            detail = normalized["category_details"][category][subitem]
            sub_score = detail["score"]
            sub_maximum = detail["max_score"]
            subitem_slot = subslot_by_key.get(subitem)
            # 짧은 버전은 LLM이 준 *_brief를 쓰고, 없으면 원문을 그대로 둔다.
            # (예전 리포트에는 _brief가 없다 — 화면에서 fallback으로 접힌다)
            assessment_brief = improvement_brief = ""
            if sub_score is None:
                assessment = "이 세부항목은 점수 자료가 없어 현재 수준을 판단할 수 없습니다."
                improvement = (
                    "점수 산정에 사용한 조직·아이디어 자료를 먼저 확인한 뒤 보완 여부를 "
                    "판단해야 합니다."
                )
            elif _is_full(sub_score, sub_maximum) or _is_full(score, maximum):
                assessment = (
                    f"현재 {sub_score}/{sub_maximum}점으로 이 세부항목은 만점입니다."
                )
                improvement = (
                    "추가 보완이 필요하지 않습니다. 현재 강점과 이를 뒷받침하는 자료를 "
                    "유지하면 됩니다."
                )
            else:
                llm_subitem = llm_category["subitems"][subitem_slot]
                assessment = _limit_text(
                    llm_subitem["assessment"], SUBITEM_ASSESSMENT_MAX_CHARS
                )
                improvement_maximum = (
                    PRIORITY_SUBITEM_IMPROVEMENT_MAX_CHARS
                    if _priority(sub_score, sub_maximum) == "high"
                    else SUBITEM_IMPROVEMENT_MAX_CHARS
                )
                improvement = _limit_text(
                    llm_subitem["improvement"], improvement_maximum
                )
                assessment_brief = _limit_text(
                    llm_subitem.get("assessment_brief"),
                    SUBITEM_ASSESSMENT_BRIEF_MAX_CHARS,
                )
                improvement_brief = _limit_text(
                    llm_subitem.get("improvement_brief"),
                    SUBITEM_IMPROVEMENT_BRIEF_MAX_CHARS,
                )
            subitems.append(
                {
                    "subitem": subitem,
                    "display_name": display_name,
                    "score": sub_score,
                    "max_score": sub_maximum,
                    "data_status": detail["data_status"],
                    "score_origin": detail["score_origin"],
                    "score_origin_label": _score_origin_label(detail["score_origin"]),
                    "assessment": assessment,
                    "assessment_brief": assessment_brief or assessment,
                    "improvement": improvement,
                    "improvement_brief": improvement_brief or improvement,
                    **_source_evidence(
                        normalized, sources, category=category, subitem=subitem
                    ),
                }
            )
        categories.append(
            {
                "category": category,
                "meaning": CATEGORY_DEFINITIONS[category]["meaning"],
                "score": score_info["score"],
                "max_score": score_info["max_score"],
                "status": score_info["status"],
                "score_origin": score_info["score_origin"],
                "score_origin_label": _score_origin_label(score_info["score_origin"]),
                "headline": _limit_text(headline, CATEGORY_HEADLINE_MAX_CHARS),
                "summary": _limit_text(summary, CATEGORY_SUMMARY_MAX_CHARS),
                "summary_brief": summary_brief
                or _limit_text(summary, CATEGORY_SUMMARY_MAX_CHARS),
                "priority": _priority(score, maximum),
                "subitem_analysis": subitems,
                **_source_evidence(normalized, sources, category=category),
            }
        )

    gaps: list[dict[str, Any]] = []
    strategies: list[dict[str, Any]] = []
    for gap_slot, gap_meta in slots["gaps"].items():
        llm_gap = payload["gaps"][gap_slot]
        strategy_set = []
        for strategy_type in STRATEGY_TYPES:
            node = payload["strategies"][gap_slot][strategy_type]
            action = str(node.get("action") or "").strip()
            rationale = str(node.get("rationale") or "").strip()
            if node["applicability"] == "not_applicable":
                action = action or "이 전략은 현재 우선 적용하지 않습니다."
                rationale = rationale or "현재 입력 기준으로 다른 확보 방식이 더 적합합니다."
            action_maximum, rationale_maximum = STRATEGY_TEXT_LIMITS[
                node["applicability"]
            ]
            action = _limit_text(action, action_maximum)
            rationale = _limit_text(rationale, rationale_maximum)
            action_brief = _limit_text(
                node.get("action_brief"), STRATEGY_ACTION_BRIEF_MAX_CHARS
            )
            rationale_brief = _limit_text(
                node.get("rationale_brief"), STRATEGY_RATIONALE_BRIEF_MAX_CHARS
            )
            strategy_set.append(
                {
                    "strategy_type": strategy_type,
                    "applicability": node["applicability"],
                    "headline": _limit_text(
                        node["headline"], STRATEGY_HEADLINE_MAX_CHARS
                    ),
                    "action": action,
                    "action_brief": action_brief or action,
                    "rationale": rationale,
                    "rationale_brief": rationale_brief or rationale,
                    **_source_evidence(normalized, sources),
                }
            )
        priority_strategy = str(llm_gap["priority_strategy"])
        priority_rationale = str(llm_gap["priority_rationale"])
        priority_rationale_brief = str(llm_gap.get("priority_rationale_brief") or "")
        strategy_by_type = {
            item["strategy_type"]: item for item in strategy_set
        }
        if strategy_by_type[priority_strategy]["applicability"] == "not_applicable":
            replacement = next(
                (
                    item
                    for applicability in ("recommended", "conditional")
                    for item in strategy_set
                    if item["applicability"] == applicability
                ),
                strategy_by_type[priority_strategy],
            )
            priority_strategy = replacement["strategy_type"]
            priority_rationale = replacement["rationale"]
            # 대체 전략으로 바뀌면 짧은 버전도 그 전략 것으로 함께 바꾼다.
            priority_rationale_brief = replacement["rationale_brief"]
        gaps.append(
            {
                **gap_meta,
                "headline": _limit_text(
                    llm_gap["headline"], GAP_HEADLINE_MAX_CHARS
                ),
                "cause": _limit_text(llm_gap["cause"], GAP_CAUSE_MAX_CHARS),
                "cause_brief": _limit_text(
                    llm_gap.get("cause_brief"), GAP_CAUSE_BRIEF_MAX_CHARS
                ) or _limit_text(llm_gap["cause"], GAP_CAUSE_MAX_CHARS),
                "impact": _limit_text(llm_gap["impact"], GAP_IMPACT_MAX_CHARS),
                "impact_brief": _limit_text(
                    llm_gap.get("impact_brief"), GAP_IMPACT_BRIEF_MAX_CHARS
                ) or _limit_text(llm_gap["impact"], GAP_IMPACT_MAX_CHARS),
                "priority_strategy": priority_strategy,
                "priority_rationale": _limit_text(
                    priority_rationale, GAP_PRIORITY_RATIONALE_MAX_CHARS
                ),
                "priority_rationale_brief": _limit_text(
                    priority_rationale_brief, GAP_PRIORITY_RATIONALE_BRIEF_MAX_CHARS
                ) or _limit_text(priority_rationale, GAP_PRIORITY_RATIONALE_MAX_CHARS),
                **_source_evidence(normalized, sources),
            }
        )
        strategies.append({"capability": gap_meta["capability"], "items": strategy_set})

    roadmap = {}
    for phase in ROADMAP_PHASES:
        node = payload["roadmap"][phase]
        actions = [
            _limit_text(item, ROADMAP_ACTION_MAX_CHARS)
            for item in node["actions"][:ROADMAP_MAX_ACTIONS]
        ]
        criteria = [
            _limit_text(item, ROADMAP_CRITERION_MAX_CHARS)
            for item in node["completion_criteria"][:ROADMAP_MAX_CRITERIA]
        ]
        roadmap[phase] = {
            "headline": _limit_text(
                node["headline"], ROADMAP_HEADLINE_MAX_CHARS
            ),
            "objective": _limit_text(
                node["objective"], ROADMAP_OBJECTIVE_MAX_CHARS
            ),
            "objective_brief": _limit_text(
                node.get("objective_brief"), ROADMAP_OBJECTIVE_BRIEF_MAX_CHARS
            ) or _limit_text(node["objective"], ROADMAP_OBJECTIVE_MAX_CHARS),
            "actions": actions,
            # 짧은 목록은 원문 목록과 같은 순서로 짝지어 쓴다. LLM이 개수를 다르게
            # 주거나 아예 빠뜨리면 그 자리만 원문으로 되돌린다(_zip_brief).
            "actions_brief": _zip_brief(
                actions, node.get("actions_brief"), ROADMAP_ACTION_BRIEF_MAX_CHARS
            ),
            "completion_criteria": criteria,
            "completion_criteria_brief": _zip_brief(
                criteria,
                node.get("completion_criteria_brief"),
                ROADMAP_CRITERION_BRIEF_MAX_CHARS,
            ),
            **_source_evidence(normalized, sources),
        }

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "diagnosis_mode": normalized["diagnosis_mode"],
        "score_contract": normalized["score_contract"],
        "generation_status": "llm_generated",
        "score_summary": copy.deepcopy(normalized["score_summary"]),
        "executive_headline": _limit_text(
            payload["executive_headline"], EXECUTIVE_HEADLINE_MAX_CHARS
        ),
        "executive_summary": _limit_text(
            payload["executive_summary"], EXECUTIVE_SUMMARY_MAX_CHARS
        ),
        "category_analysis": categories,
        "gap_analysis": gaps,
        "strategies": strategies,
        "roadmap": roadmap,
        "unverified_items": [
            _limit_text(item, UNVERIFIED_ITEM_MAX_CHARS)
            for item in list(payload.get("unverified_items") or [])[:8]
        ],
        "warnings": list(normalized["warnings"]),
        "sources": list(sources),
        "context": copy.deepcopy(normalized["context"]),
    }
    report["dashboard_text"] = _build_legacy_dashboard_text(report)
    return report


def _limit_text(value: Any, maximum: int) -> str:
    """긴 LLM 문장을 가능한 한 문장 끝에서 잘라 화면 최대 길이를 지킨다."""

    text = " ".join(str(value or "").split())
    if len(text) <= maximum:
        return text
    clipped = text[:maximum]
    sentence_end = max(
        clipped.rfind("다."),
        clipped.rfind("요."),
        clipped.rfind("니다."),
        clipped.rfind("한다."),
    )
    if sentence_end >= int(maximum * 0.7):
        return clipped[: sentence_end + 2].strip()
    return clipped[: maximum - 1].rstrip(" ,;:") + "…"


def _zip_brief(full: list[str], brief: Any, maximum: int) -> list[str]:
    """원문 목록과 짧은 목록을 순서대로 짝지어 대시보드용 목록을 만든다.

    LLM이 개수를 다르게 주거나 항목을 비워 보낼 수 있으므로, 짝이 없거나 빈
    자리는 원문을 그대로 쓴다 — 화면에서 항목이 사라지는 것보다 낫다.
    """
    items = brief if isinstance(brief, list) else []
    result = []
    for index, original in enumerate(full):
        candidate = items[index] if index < len(items) else ""
        result.append(_limit_text(candidate, maximum) or original)
    return result


def _basis_label(basis: str) -> str:
    return {
        "score_input": "점수 엔진 입력",
        "organization_data": "조직 데이터 근거",
        "idea_data": "아이디어·DB 데이터 근거",
        "mixed": "입력 근거 + LLM 해석·권고",
        "llm_recommendation": "LLM 추정·권고",
    }.get(basis, "LLM 추정·권고")


def _score_origin_label(origin: str) -> str:
    return {
        "organization_score_engine": "조직 데이터 기반 55점 점수엔진",
        "database_score_engine": "신사업 DB + 공통 점수엔진",
        "llm_estimate_plus_score_engine": "LLM 시장 추정 + 공통 점수엔진",
    }.get(origin, "점수 산정 근거 확인 필요")


def _build_no_gap_report(
    normalized: Mapping[str, Any], sources: Sequence[Mapping[str, str]]
) -> dict[str, Any]:
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "diagnosis_mode": normalized["diagnosis_mode"],
        "score_contract": normalized["score_contract"],
        "generation_status": "no_gap",
        "score_summary": copy.deepcopy(normalized["score_summary"]),
        "executive_headline": "현재 점수 결과에서는 추가로 진단할 부족 역량이 확인되지 않았습니다.",
        "executive_summary": "앞 단계의 역량 비교에서 부족 역량이 확인되지 않았습니다.",
        "category_analysis": [],
        "gap_analysis": [],
        "strategies": [],
        "roadmap": {},
        "unverified_items": [],
        "warnings": list(normalized["warnings"]),
        "sources": list(sources),
        "context": copy.deepcopy(normalized["context"]),
    }
    report["dashboard_text"] = report["executive_summary"]
    return report


def _build_legacy_dashboard_text(report: Mapping[str, Any]) -> str:
    lines = [report.get("executive_summary", ""), "", "[평가항목 종합 분석]"]
    for item in report.get("category_analysis", []):
        score = "미평가" if item["score"] is None else f"{item['score']}/{item['max_score']}점"
        lines.append(f"- {item['category']} ({score}): {item['summary']}")
    lines.extend(["", "[갭 리포트]"])
    for item in report.get("gap_analysis", []):
        lines.append(
            f"- {item['capability']}: {item['cause']} 영향: {item['impact']} "
            f"우선 확보 방향: {item['priority_strategy']} - "
            f"{item['priority_rationale']} ({item['basis_label']})"
        )
    lines.extend(["", "[실행 로드맵]"])
    for phase, label in (
        ("short_term", "단기"),
        ("mid_term", "중기"),
        ("long_term", "장기"),
    ):
        node = report.get("roadmap", {}).get(phase)
        if node:
            lines.append(f"- {label}: {node['objective']}")
    return "\n".join(lines).strip()


def _optional_number(value: Any, path: str) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise F5InputError(f"{path}는 숫자여야 합니다.") from exc
    number = float(value)
    if not math.isfinite(number):
        raise F5InputError(f"{path}는 유한한 숫자여야 합니다.")
    return number


def _tidy(value: float | None) -> int | float | None:
    if value is None:
        return None
    return int(value) if float(value).is_integer() else round(float(value), 2)


def _json_safe_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, item in value.items():
        if item is None or isinstance(item, (str, int, float, bool)):
            safe[str(key)] = item
        elif isinstance(item, Mapping):
            safe[str(key)] = _json_safe_mapping(item)
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            safe[str(key)] = [str(x) for x in item]
        else:
            safe[str(key)] = str(item)
    return safe


__all__ = [
    "CATEGORY_DEFINITIONS",
    "DEFAULT_MODEL",
    "DIAGNOSIS_MODE_IDEA_FIT",
    "DIAGNOSIS_MODE_RECOMMENDATION",
    "F5ConfigurationError",
    "F5Error",
    "F5InputError",
    "F5ProviderError",
    "F5ResponseError",
    "REPORT_SCHEMA_VERSION",
    "SCORE_SCHEMA_8_VERSION",
    "SCORE_SCHEMA_IDEA_FIT_VERSION",
    "SCORE_SCHEMA_RECOMMENDATION_VERSION",
    "build_dashboard_view",
    "export_gap_report_pdf",
    "generate_gap_report",
]
