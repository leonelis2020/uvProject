"""
UV 모사 팔레트 연구실 (UV Simulation Palette Lab)
==================================================
Fat-Client (Streamlit) + Thin-DB (Supabase) 스카폴딩.

설계 원칙 (v2 — UI 개편 + GPT-4o-mini 전환 반영)
---------------------------------------------------
* 5x10 명채도 매트릭스 타일과 같은 휘발성/연산 집약적 데이터는 DB에 저장하지 않고,
  프론트엔드(Streamlit)에서 실시간 연산한다. DB는 `matrix_json`(3x4) + 대표 색상
  `base_colors`(5 HEX)만 보관한다.
* RAG 컨텍스트는 `standard_illuminants.spd_data` 와 `reflectance_library.spectral_values`
  두 마스터 테이블에서 조회한다. **GPT-4o-mini 호출 시 이미지는 전달하지 않으며**
  오직 분광/SPD JSON 만 보낸다 (토큰 절약 + 환각 방지).
* 모든 사용자 산출물(`projects`)은 덮어쓰지 않고 `parent_id` 셀프 참조 스냅샷을 통해
  버전 히스토리를 유지한다. 스크랩(reuse)은 익명 카운트(`scrap_count`) 만 누적한다.
* 50칸 팔레트는 OKLCH 색공간에서 명도(L) 만 변주하여 클라이언트에서 재계산한다.

이 파일은 "큰 틀(Scaffolding)"이며, 각 단계(PHASE 1~5)는 향후 세부 알고리즘
(OpenCV 변환, drawable-canvas 마스킹, GPT-4o-mini 호출 등)이 살을 붙일 수
있도록 추상화된 함수와 명확히 구분된 섹션 주석으로 정리되어 있다.
"""

from __future__ import annotations

import io
import json
import os
import uuid
from datetime import datetime, timedelta
from typing import Any

import streamlit as st
from st_supabase_connection import SupabaseConnection

# 이미지/마스크 처리 — 멀티리전 변환 합성에 필요
# (런타임에 실제 사용되는 함수 내부에서 import 해도 되지만 모듈 최상단에 모아둔다)
try:
    import numpy as np
    from PIL import Image, ImageOps
except Exception:  # pragma: no cover - 의존성 미설치 시 import 가드
    np = None  # type: ignore
    Image = None  # type: ignore
    ImageOps = None  # type: ignore

try:
    import cv2  # OpenCV — 마스크 리사이즈(INTER_NEAREST) 전용
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore


# ── set_page_config 는 모든 Streamlit 명령보다 먼저 실행돼야 한다 ─────────────
# 아래의 `conn = st.connection(...)` / `st.error(...)` 등 모듈 레벨 Streamlit
# 호출이 있어서, `main()` 안에 두면 "set_page_config can only be called once
# per app" 에러가 나거나 새 codespace 에서 앱이 안 열리는 일이 있다 (사용자
# 보고). 모듈 최상단으로 끌어올린다.
st.set_page_config(
    page_title="UV 모사 팔레트 연구실",
    page_icon="🎨",
    layout="wide",
)


# ── Streamlit 버전 호환 헬퍼 ──────────────────────────────────────────────────
# Streamlit 의 width 관련 API 가 두 번 deprecate 됐다:
#   * `use_column_width=True`     : ~1.32 이전 표준 → 1.32+ 에서 deprecated
#   * `use_container_width=True`  : 1.32 표준     → 1.51+ 에서 deprecated
#   * `width="stretch"|"content"|int` : 1.51+ 신표준
# 핀이 `>=1.36,<1.56` 이라 두 deprecation 모두에 걸칠 수 있어
# 런타임 버전 체크로 올바른 kwargs 를 반환한다.
def _full_width_kwargs() -> dict:
    try:
        ver = tuple(int(x) for x in st.__version__.split(".")[:2])
    except Exception:
        ver = (0, 0)
    if ver >= (1, 51):
        return {"width": "stretch"}
    return {"use_container_width": True}


_FW = _full_width_kwargs()


# ── streamlit-drawable-canvas 호환성 정책 ─────────────────────────────────────
# `streamlit-drawable-canvas == 0.9.3` 는 Streamlit 1.42 직전 기준으로 작성됐고,
# 그 이후 Streamlit 이 `image_to_url` 의 시그니처(`width: int` → `layout_config`)
# 를 바꿔서 두 가지 패턴 모두 깨진다:
#   (a) 함수가 모듈에서 사라져 `AttributeError: ... has no attribute 'image_to_url'`
#   (b) 함수는 찾아도 시그니처 불일치로 `AttributeError: 'int' has no attribute 'width'`
# 이전 버전의 monkey-patch 는 (a) 만 막아주고 (b) 를 그대로 노출시키는 부작용이
# 있었다 (시연 화면 → 모든 태그 클릭 시 트레이스백 + UI 흔들림).
#
# 결론: monkey-patch 를 **제거**하고, `_render_mask_canvas_for_region` 에서
# `st_canvas` 호출을 **모든 예외에 대해** 받아 bbox 슬라이더 폴백으로 전환한다.
# 마우스 freedraw 가 필요해지면 향후 별도 컴포넌트(custom HTML canvas 등) 로
# 대체한다. 현재 UX 는 사각 영역 마스킹 + 멀티리전 합성으로 충분히 작동한다.

# =============================================================================
# PHASE 0: CONNECTION & MASTER LAYER
#   - 시스템 환경 변수 → secrets.toml 백업 순으로 자격 증명 로드 (기존 로직 보호)
#   - `standard_illuminants`, `reflectance_library` 마스터 데이터 접근 헬퍼
# =============================================================================

s_url = os.environ.get("SUPABASE_URL")
s_key = os.environ.get("SUPABASE_KEY")

# 만약 환경 변수가 없으면 secrets.toml에서 가져오도록 백업 로직
if not s_url:
    try:
        s_url = st.secrets["connections"]["supabase"]["url"]
        s_key = st.secrets["connections"]["supabase"]["key"]
    except Exception:
        st.error("Supabase 연결 설정(Secrets 또는 환경변수)이 필요합니다!")

conn = st.connection(
    "supabase",
    type=SupabaseConnection,
    url=s_url,
    key=s_key,
)


def _get_openai_key() -> str | None:
    """OpenAI API 키 하이브리드 조회 — Supabase 자격 증명과 동일한 정책.

    1순위: 환경 변수 `OPENAI_API_KEY` (배포 환경 / Codespaces 시크릿)
    2순위: `.streamlit/secrets.toml` 의 `[openai] api_key = "..."`

    `_openai_key_info()` 는 같은 우선순위를 따르되 어디서 키가 왔는지
    + 키의 접두/접미만 추출해 진단용으로 반환한다.
    """
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    try:
        return st.secrets["openai"]["api_key"]  # type: ignore[index]
    except Exception:
        return None


def _format_openai_error_hint(err_str: str | None) -> str | None:
    """OpenAI 호출 실패 메시지를 사용자가 바로 행동할 수 있는 한 줄로 변환.

    설정 모달의 진단 패널이 `detail` 필드를 그대로 보여주는 것에 더해, 자주
    걸리는 패턴(잔액 부족 / 키 무효 / 레이트 리밋) 일 때만 따로 강조해 준다.
    매칭 안 되면 `None` 을 반환해 힌트를 표시하지 않는다.
    """
    if not err_str:
        return None

    # OpenAI 잔액 부족 (insufficient_quota) — 429 와 함께 옴.
    # rate_limit (단순 분당 제한) 과 분리해서 안내해야 한다: 결제 추가 vs 대기.
    if "insufficient_quota" in err_str:
        return (
            "💳 OpenAI 계정 **크레딧 부족** (insufficient_quota).\n"
            "https://platform.openai.com/settings/organization/billing 에서 "
            "결제 수단을 추가하고 **최소 $5 크레딧 충전** 이 필요합니다. "
            "(현재 Usage 페이지에 Total Spend $0.00 이 보이는 이유이며, "
            "API 호출이 quota 검사 단계에서 차단되어 모델까지 도달하지 못함)"
        )
    if "invalid_api_key" in err_str or "Incorrect API key" in err_str:
        return (
            "🔑 API 키가 **유효하지 않음** — `.streamlit/secrets.toml` 의 "
            "`[openai] api_key` 가 올바른 `sk-...` 형식인지 다시 확인하세요."
        )
    # insufficient_quota 와 분리: 잔액은 있으나 분당 호출량 초과.
    if "rate_limit" in err_str.lower() and "insufficient_quota" not in err_str:
        return "⏳ OpenAI **레이트 리밋** — 잠시 후 다시 시도하세요."
    if "401" in err_str:
        return "🔐 인증 실패 (401) — API 키가 만료/취소되었거나 다른 조직 키입니다."
    return None


def _openai_key_info() -> dict:
    """진단용 — 키의 출처(env/secrets/없음) + 길이 + 마스킹된 미리보기."""
    src = None
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        src = "env"
    else:
        try:
            key = st.secrets["openai"]["api_key"]  # type: ignore[index]
            src = "secrets"
        except Exception:
            key = None
    if not key:
        return {"source": None, "length": 0, "preview": None}
    preview = f"{key[:6]}…{key[-4:]}" if len(key) > 10 else "***"
    return {"source": src, "length": len(key), "preview": preview}


# 마지막 OpenAI 호출 진단 — 설정 모달의 'AI 진단' 섹션이 표시
SS_OPENAI_LAST_DIAG = "openai_last_diag"


# -----------------------------------------------------------------------------
# 마스터 데이터 헬퍼 (RAG context loaders)
# -----------------------------------------------------------------------------
def get_reflectance_data(label: str) -> list[dict] | None:
    """RAG용 반사율 라이브러리 조회 (기존 시그니처 보호)."""
    query = (
        conn.table("reflectance_library")
        .select("*")
        .eq("label", label)
        .execute()
    )
    return query.data if query.data else None


def get_illuminant_spd(name: str) -> dict | None:
    """광원(`standard_illuminants`) SPD(분광 전력 분포) 단일 조회."""
    try:
        query = (
            conn.table("standard_illuminants")
            .select("*")
            .eq("name", name)
            .single()
            .execute()
        )
        return query.data if getattr(query, "data", None) else None
    except Exception:
        return None


@st.cache_data(ttl=300, show_spinner=False)
def list_subjects() -> list[dict]:
    """피사체 50종 전체 목록 (중앙 태그 탭에서 사용).

    [버그픽스] false-color 변환은 spectral_values(분광 반사율) 가 반드시 필요한데,
    과거엔 SELECT 에서 빠져 있어 변환이 무력화됐다(AI 방식은 name 만 써서 안 걸림).
    spectral_values 를 함께 가져온다.
    """
    try:
        resp = (
            conn.table("reflectance_library")
            .select("id, category, label, is_uv_active, spectral_values")
            .order("category")
            .order("label")
            .execute()
        )
        return resp.data or []
    except Exception:
        return []


@st.cache_data(ttl=300, show_spinner=False)
def list_illuminants() -> list[dict]:
    """광원 5종 전체 목록.

    [버그픽스] false-color 변환은 spd_data(분광 전력 분포) 가 반드시 필요한데,
    과거엔 SELECT 에서 빠져 있어 선택된 광원에 spd_data 가 없어 변환이 무력화됐다.
    spd_data 를 함께 가져온다.
    """
    try:
        resp = (
            conn.table("standard_illuminants")
            .select("id, name, description, spd_data")
            .order("name")
            .execute()
        )
        return resp.data or []
    except Exception:
        return []


# =============================================================================
# session_state 키 표준 (모듈 간 계약)
# =============================================================================
SS_ORIGIN_IMAGE = "origin_image_pil"        # PIL.Image — 1024px 리사이즈 본
SS_ORIGIN_BYTES = "origin_image_bytes"      # bytes — Storage 업로드용 원본
SS_RESULT_BYTES = "result_image_bytes"      # bytes — 변환 결과
SS_MATRIX = "matrix_json"                   # dict — GPT-4o-mini 가 반환한 3x4 매트릭스
SS_BASE_COLORS = "base_colors"              # list[str] — HEX 5개
SS_TILE_GRID = "tile_grid_5x10"             # list[list[str]] — 휘발성, DB 미저장
SS_IS_ANON = "is_anon"                      # bool — 익명 닉네임 (is_anonymous)
SS_IS_PALETTE_PUBLIC = "is_palette_public"  # bool — 팔레트 공개 여부
SS_AUTH_USER = "auth_user"                  # dict | None — Supabase Auth user
SS_SELECTED_ILLU = "selected_illu"          # dict | None — standard_illuminants row (단일)
# ── 멀티리전(영역별 피사체) ─────────────────────────────────────────────
# 광원은 이미지 전체 1개로 고정 (사진 한 장은 한 조명 아래 찍힌 것).
# 피사체는 영역마다 N개 → region 리스트로 관리.
#   region = {
#     "subject": dict,             # reflectance_library row (label/category/spectral_values/...)
#     "mask":    np.ndarray|None,  # H×W bool. None 이면 전체 적용
#     "matrix":  dict|None,        # Convert 직후 채워짐 (3×4)
#   }
SS_REGIONS = "regions"


# =============================================================================
# PHASE 1: INPUT  —  이미지 업로드 / 리사이즈
#   - PIL.Image.thumbnail((1024, 1024)) 1024px 리사이즈
#   - 결과는 st.session_state 에 저장하여 PHASE 3 가 소비한다.
#   (drawable-canvas 마스킹은 현재 UI 안에는 노출되지 않으며 향후 옵션 슬롯으로 유지)
# =============================================================================

def handle_image_upload(uploaded_file) -> None:
    """업로드 파일을 1024px 썸네일로 리사이즈하고 session_state 에 저장.

    canvas 배경 이미지로 사용되며, 멀티리전 변환의 좌표 기준이 된다.
    EXIF 회전을 먼저 보정하지 않으면 마스크 좌표가 어긋나므로 `exif_transpose` 필수.
    """
    if Image is None:
        raise NotImplementedError("Pillow 미설치 — requirements 설치 필요")
    img = Image.open(uploaded_file).convert("RGB")
    if ImageOps is not None:
        img = ImageOps.exif_transpose(img)
    img.thumbnail((1024, 1024))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    st.session_state[SS_ORIGIN_IMAGE] = img
    st.session_state[SS_ORIGIN_BYTES] = buf.getvalue()


# =============================================================================
# PHASE 2-SPECTRAL: UV FALSE-COLOR 변환 코어 (AI/GPT 제거 · 순수 numpy)
# -----------------------------------------------------------------------------
#   설계 변경 (v3 — GPT 폐기):
#     기존엔 GPT-4o-mini 가 광원 '이름' 만 보고 3x4 행렬을 '추측' 했다. 그 결과
#     SPD 숫자값이 결과에 반영되지 않고, UV-A 가 무조건 보라색으로 하드코딩됐다.
#
#     이제는 GPT 를 호출하지 않고, 분광 반사율 R(λ) × 광원 SPD S(λ) 를 직접
#     적분해 4개 대역(UV/Blue/Green/Red) 신호를 만들고, 이를 가시광 3채널로
#     시프트 매핑(UV→B, B→G, G→R)한다 → "false-color UV" (네온 형광이 아니라
#     반사율 기반 무지개). 광원 5종의 SPD 차이가 결과를 물리적으로 결정한다.
#
#     - UVA_365 처럼 가시광 에너지가 0 인 블랙라이트 → UV 대역만 살아 거의 단색
#       파랑. (형광이 없으면 깜깜한 게 물리적으로 옳다.)
#     - D65/Halogen/LED/F11 은 SPD 모양에 따라 각각 다른 색조로 갈린다.
#
#   기존 함수(call_gpt4o_*, _MATRIX_SYSTEM_PROMPT, apply_matrix_array 등)는
#   하위 호환을 위해 정의만 남겨두되, 변환 경로(_run_convert_multi)에서는 더 이상
#   호출하지 않는다.
# =============================================================================

# 41포인트 파장 그리드 (300–700nm, 10nm) — spectral_values / spd_data 와 동일 격자
WAVELENGTHS = list(range(300, 701, 10))


def spectrum_to_array(spec: dict | None, fill: float = 0.0) -> "np.ndarray":
    """{"300": 0.08, ...} (str/int 키 혼용 허용) → 41-길이 float 배열. 누락은 fill.

    Supabase jsonb 컬럼이 dict 가 아니라 JSON '문자열' 로 반환되는 환경이 있어
    (이게 'D65/블랙라이트 골라도 안 변함' 의 원인이었다) 문자열이면 먼저 파싱한다.
    """
    if np is None:
        raise NotImplementedError("numpy 미설치")
    # jsonb 가 문자열로 온 경우 파싱
    if isinstance(spec, str):
        try:
            spec = json.loads(spec)
        except Exception:
            spec = None
    out = np.full(len(WAVELENGTHS), fill, dtype=np.float64)
    if not spec or not isinstance(spec, dict):
        return out
    for i, wl in enumerate(WAVELENGTHS):
        v = spec.get(str(wl), spec.get(wl))
        if v is not None:
            try:
                out[i] = float(v)
            except (TypeError, ValueError):
                pass
    return out


def _band_masks() -> dict:
    """UV / Blue / Green / Red 4대역의 파장 인덱스 마스크."""
    wl = np.array(WAVELENGTHS)
    return {
        "UV":    (wl >= 300) & (wl < 400),   # 300–390
        "BLUE":  (wl >= 400) & (wl < 500),   # 400–490
        "GREEN": (wl >= 500) & (wl < 600),   # 500–590
        "RED":   (wl >= 600) & (wl <= 700),  # 600–700
    }


def compute_false_color_target(
    reflectance: dict | None,
    spd: dict | None,
    is_uv_active: bool = False,
) -> dict:
    """피사체 1종 × 광원 1개 → false-color 기준 RGB(0–1) + 진단.

    매핑:  Blue_out ← UV,  Green_out ← Blue,  Red_out ← Green  (Red 대역은 약하게만)
    광원 정규화: 가시광(400–700) 총에너지를 1로 → 밝기차 제거, 색조만 비교.
      (UVA_365 처럼 가시광 0 인 광원은 정규화되지 않아 UV 대역만 살아남는다.)
    """
    if np is None:
        raise NotImplementedError("numpy 미설치")
    R = spectrum_to_array(reflectance, fill=0.0)
    S = spectrum_to_array(spd, fill=0.0)

    wl = np.array(WAVELENGTHS)
    vis = wl >= 400
    vis_energy = S[vis].sum()
    if vis_energy > 1e-9:
        S = S / vis_energy

    reflected = R * S
    masks = _band_masks()

    def band(m):
        sel = reflected[m]
        return float(sel.mean()) if sel.size else 0.0

    bands = {name: band(m) for name, m in masks.items()}

    r_out = bands["GREEN"] + bands["RED"] * 0.25
    g_out = bands["BLUE"]
    b_out = bands["UV"]

    # 형광 피사체가 UV-only 환경일 때만 녹색 끼 살짝 (emission 컬럼 생기면 교체)
    if is_uv_active and bands["UV"] > 0 and (bands["BLUE"] + bands["GREEN"]) < 1e-6:
        g_out += bands["UV"] * 0.15

    rgb = np.array([r_out, g_out, b_out], dtype=np.float64)
    peak = rgb.max()
    if peak > 1e-9:
        rgb = rgb / peak  # 색조 유지 정규화 (밝기는 apply 단계서 원본 luma 로 입힘)
    rgb = np.clip(rgb, 0.0, 1.0)

    total = sum(bands.values()) or 1e-9
    return {"rgb": rgb.tolist(), "bands": bands, "uv_fraction": bands["UV"] / total}


def apply_false_color_array(
    img: "np.ndarray",
    target_rgb,
    strength: float = 1.0,
    mode: str = "iridescent",
    sat_boost: float = 7.0,
) -> "np.ndarray":
    """H×W×3 [0,1] 원본을 false-color 로 변환. 두 가지 모드.

    mode="flat":
        원본 luma(밝기) × 단일 target 색. 영역이 한 색으로 균일하게 물든다.
        (광원 색조를 평평하게 입히고 싶을 때.)

    mode="iridescent" (기본):
        '무지개 까마귀' 모드. 원본 픽셀의 미세한 색편차(구조색 광택)를 분리·증폭해
        부위마다 파랑/보라/청록이 다르게 나타나는 영롱한 결과를 만든다. 까마귀처럼
        검은 깃털의 미묘한 광택이 위치에 따라 다른 색으로 펼쳐진다.
        sat_boost 가 클수록 무지개가 강해진다.

    멀티리전 합성에서 영역마다 target_rgb / strength / mode 를 바꿔 반복 호출한다.
    """
    if np is None:
        raise NotImplementedError("numpy 미설치")
    target = np.asarray(target_rgb, dtype=np.float32).reshape(1, 1, 3)
    luma = (
        0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
    )[..., None]

    if mode == "flat":
        recolored = np.clip(luma * target * 1.6, 0.0, 1.0)
    else:  # iridescent
        # 1) 무채색(밝기) 제거 → 픽셀별 미세 색편차(구조색)만 남김
        chroma = img - luma
        # 2) 색편차 증폭 → 미묘한 광택을 큰 색차로
        amp = chroma * sat_boost
        # 3) false-color 축으로 채널 시프트 (UV쪽 = 파랑/보라/청록 계열로)
        #    원래 G편차→R, B편차→G, R편차→B  (가시 구조색을 UV 무지개로 회전)
        irid = np.stack([amp[..., 1], amp[..., 2], amp[..., 0]], axis=-1)
        # 4) '베이스 + 광택':
        #    - 베이스: 원본 밝기에 target 색조를 입히되 너무 어두워지지 않게.
        #      target 색조(파랑)만 곱하면 칙칙해지므로 약간의 중성 밝기를 섞는다.
        #    - 광택: 밝은 부위(luma 큰 곳)에서만 강하게 드러나도록 luma 로 게이팅
        #      → 어두운 그림자엔 광택이 안 끼고, 빛 받는 면에서만 무지개가 번쩍.
        tinted = luma * target              # 색조 입힌 밝기
        neutral = luma * 0.5               # 중성 밝기 (칙칙함 방지)
        base = (tinted * 0.7 + neutral * 0.3) * 1.25
        gate = np.clip(luma * 1.3, 0.0, 1.0)
        recolored = np.clip(base + irid * gate * 0.5, 0.0, 1.0)

    out = img * (1.0 - strength) + recolored * strength
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# =============================================================================
# PHASE 2: AI SIMULATION  —  [DEPRECATED · 미사용]
#   ↓ 아래 GPT-4o-mini 경로는 v3 에서 변환에 더 이상 사용되지 않는다.
#     함수 정의는 하위 호환(외부 import) 을 위해 남겨두지만, _run_convert_multi
#     는 compute_false_color_target / apply_false_color_array 만 호출한다.
# =============================================================================

EXPECTED_MATRIX_KEYS = {"r", "g", "b"}
EXPECTED_MATRIX_ROW_LEN = 4  # [r0, r1, r2, bias]


def build_rag_context(subject: dict, illuminant: dict) -> dict:
    """GPT-4o-mini 프롬프트에 결합할 RAG 컨텍스트 패키지 (이미지 X, 분광만).

    Args:
        subject:    `reflectance_library` row (spectral_values, is_uv_active, ...)
        illuminant: `standard_illuminants` row (spd_data, name, ...)
    """
    return {
        "subject": {
            "label": subject.get("label"),
            "category": subject.get("category"),
            "is_uv_active": subject.get("is_uv_active"),
            "spectral_values": subject.get("spectral_values"),  # 300–700nm, 41pt, 0–1
        },
        "illuminant": {
            "name": illuminant.get("name"),
            "spd_data": illuminant.get("spd_data"),
        },
    }


def validate_matrix_payload(payload: Any) -> bool:
    """GPT-4o-mini 가 반환한 JSON 이 3x4 매트릭스 규약을 따르는지 검증."""
    if not isinstance(payload, dict):
        return False
    if not EXPECTED_MATRIX_KEYS.issubset(payload.keys()):
        return False
    for ch in EXPECTED_MATRIX_KEYS:
        row = payload.get(ch)
        if not isinstance(row, list) or len(row) != EXPECTED_MATRIX_ROW_LEN:
            return False
        if not all(isinstance(v, (int, float)) for v in row):
            return False
    return True


_MATRIX_SYSTEM_PROMPT = (
    "You are a color-science assistant performing UV / non-standard illumination "
    "photography simulation. Input per request:\n"
    "- subject: spectral reflectance over 300–700 nm (~41 points, 0–1), plus an "
    "  `is_uv_active` flag — TRUE means the subject fluoresces under UV light.\n"
    "- illuminant: spectral power distribution (SPD) of the light source plus a "
    "  human-readable `name` such as 'UV-A', 'Sunrise', 'D65'.\n\n"
    "Compute a 3x4 affine RGB matrix M that, when applied as "
    "`out = M @ [r, g, b, 1.0]` to each pixel of an ordinary daylight photo of the "
    "subject, SIMULATES how it would appear under the supplied illuminant.\n\n"
    "Concrete scenario behaviour you MUST follow:\n\n"
    "1. UV-A 'black light' (illuminant.name == 'UV-A' OR SPD dominated by 300–400 nm "
    "   with little 400–700 nm energy):\n"
    "   * Drastically REDUCE the visible-band response — diagonals around 0.10–0.35, "
    "     NOT 1.0. The scene should look much darker than daylight.\n"
    "   * Apply a strong VIOLET / DEEP-BLUE cast: blue channel coefficients should "
    "     dominate (b row roughly [0.10, 0.15, 0.85, 0.10]) and red/green should "
    "     contribute partial blue via cross-channel terms (r row roughly "
    "     [0.20, 0.05, 0.30, 0.05], g row roughly [0.05, 0.20, 0.25, 0.05]).\n"
    "   * If subject is_uv_active=TRUE, add a positive bias to whichever visible "
    "     channel approximates the subject's fluorescence emission (typically GREEN "
    "     or RED bias around +0.10–0.20) so painted regions visibly POP.\n\n"
    "2. Sunrise / Sunset (warm SPD heavy in 580–700 nm):\n"
    "   * Boost red, slightly boost green, suppress blue. Example: r ≈ "
    "     [1.25, 0.10, 0.00, 0.08], g ≈ [0.05, 0.95, 0.00, 0.02], b ≈ "
    "     [0.00, 0.00, 0.55, -0.05].\n\n"
    "3. D65 standard daylight: stays close to identity but slightly cooler — "
    "   diagonals near 0.95–1.05, not exactly 1.0.\n\n"
    "HARD RULES (violation = failure):\n"
    "- You MUST NOT return the exact identity matrix "
    '{"r":[1,0,0,0],"g":[0,1,0,0],"b":[0,0,1,0]}. That value is reserved as the '
    "calling code's 'unknown' fallback and would render the original photo "
    "unchanged.\n"
    "- Make a physically-motivated estimate even with limited data. Bias toward a "
    "visibly noticeable transformation: a slightly wrong but visible result is "
    "preferable to a perfect-but-invisible one.\n\n"
    "Return STRICT JSON ONLY (no prose, no markdown fences) with this exact shape:\n"
    '{"r":[r0,r1,r2,bias],"g":[g0,g1,g2,bias],"b":[b0,b1,b2,bias]}\n\n'
    "All twelve values are floats in [-2.0, 2.0]."
)


# 결정론적 illuminant 폴백 — AI 가 identity 만 뱉을 때 시각적 변화를 보장하기 위함.
# 이름을 lower() 비교하므로 'uv-a', 'UV-A', 'uva' 모두 매칭.
_ILLUMINANT_FALLBACK_MATRICES: dict[str, dict] = {
    "uv-a": {
        # Black light 가상 — 어둡고 보랏빛, 마젠타 풍의 결과.
        "r": [0.20, 0.05, 0.30, 0.05],
        "g": [0.05, 0.20, 0.25, 0.05],
        "b": [0.10, 0.15, 0.85, 0.10],
    },
    "sunrise": {
        # 일출 — 따뜻한 오렌지 캐스트.
        "r": [1.25, 0.10, 0.00, 0.08],
        "g": [0.05, 0.95, 0.00, 0.02],
        "b": [0.00, 0.00, 0.55, -0.05],
    },
    "sunset": {
        # 일몰 — 일출보다 더 붉음.
        "r": [1.35, 0.12, 0.00, 0.12],
        "g": [0.08, 0.88, 0.00, 0.00],
        "b": [0.00, 0.00, 0.45, -0.10],
    },
    "d65": {
        # 표준 주광 — 거의 동등하지만 살짝 cooler. (완벽한 항등을 피한다)
        "r": [0.95, 0.00, 0.05, 0.00],
        "g": [0.00, 1.00, 0.00, 0.00],
        "b": [0.00, 0.00, 1.05, 0.00],
    },
}


def _illuminant_fallback(illu_name: str | None) -> dict:
    """illuminant 이름 기반 결정론적 행렬. 매칭 실패 시 약한 violet bias 반환.

    `call_gpt4o_mini_for_matrix` 의 최종 폴백 — IDENTITY 를 직접 반환하지 않고
    이걸 호출하면 사용자에게 '아무것도 안 바뀜' 보다는 시각적 단서가 남는다.
    """
    key = (illu_name or "").strip().lower()
    if key in _ILLUMINANT_FALLBACK_MATRICES:
        return dict(_ILLUMINANT_FALLBACK_MATRICES[key])
    # 알 수 없는 illuminant — 미약한 cool/violet 캐스트 (식별 가능한 변화)
    return {
        "r": [0.85, 0.00, 0.10, 0.00],
        "g": [0.00, 0.90, 0.05, 0.00],
        "b": [0.05, 0.05, 1.00, 0.05],
    }


def _is_exact_identity(payload: dict) -> bool:
    """payload 가 정확히 IDENTITY 행렬인지 (AI 가 hedging 으로 안전한 답을
    내놓은 케이스를 거르기 위함)."""
    try:
        return (
            list(payload.get("r", [])) == [1.0, 0.0, 0.0, 0.0]
            and list(payload.get("g", [])) == [0.0, 1.0, 0.0, 0.0]
            and list(payload.get("b", [])) == [0.0, 0.0, 1.0, 0.0]
        )
    except Exception:
        return False


def _record_openai_diag(**kv) -> None:
    """OpenAI 호출 직후 진단 상태를 session_state 에 적재. 설정 모달이 표시."""
    diag = st.session_state.get(SS_OPENAI_LAST_DIAG) or {}
    diag.update(kv)
    diag["ts"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    st.session_state[SS_OPENAI_LAST_DIAG] = diag


def call_gpt4o_mini_for_matrix(
    rag_context: dict,
    *,
    max_retries: int = 2,
    temperature: float = 0.3,
) -> dict:
    """GPT-4o-mini 호출 → 3x4 변환 행렬 JSON. **이미지는 전달하지 않는다.**

    Flow:
      1. env-var → secrets.toml 하이브리드로 OPENAI_API_KEY 조회.
      2. RAG context 를 `_MATRIX_SYSTEM_PROMPT` + JSON 사용자 메시지로 합성.
      3. `response_format={"type":"json_object"}` 로 강제 JSON 응답.
      4. `validate_matrix_payload()` 통과 시 즉시 반환.
      5. 실패 시 최대 `max_retries` 회 재시도.
      6. 끝까지 실패하면 IDENTITY 를 반환 (예외를 던지지 않음 →
         `_run_convert_multi` 의 부분 실패 격리 정책과 일관).

    [진단성]
      모든 분기에서 `_record_openai_diag(...)` 로 마지막 호출의 상태를 저장한다.
      "API 키 넣었는데 OpenAI 대시보드에 호출이 안 잡힘" 같은 시연 보고를
      빠르게 식별할 수 있도록 설정 모달의 AI 진단 패널에 노출된다.
    """
    api_key = _get_openai_key()
    if not api_key:
        _record_openai_diag(
            stage="no_api_key",
            ok=False,
            detail=(
                "OPENAI_API_KEY 환경변수도 없고 .streamlit/secrets.toml 의 "
                "[openai] api_key 도 없습니다."
            ),
        )
        return dict(IDENTITY)

    try:
        from openai import OpenAI
    except Exception as e:
        _record_openai_diag(
            stage="openai_import_failed",
            ok=False,
            detail=f"openai 패키지 import 실패: {type(e).__name__}: {e}",
        )
        return dict(IDENTITY)

    try:
        client = OpenAI(api_key=api_key)
    except Exception as e:
        _record_openai_diag(
            stage="client_init_failed",
            ok=False,
            detail=f"OpenAI 클라이언트 초기화 실패: {type(e).__name__}: {e}",
        )
        return dict(IDENTITY)

    user_payload = {
        "subject": rag_context.get("subject"),
        "illuminant": rag_context.get("illuminant"),
    }

    last_err: str | None = None
    last_raw: str | None = None
    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                response_format={"type": "json_object"},
                temperature=temperature,
                messages=[
                    {"role": "system", "content": _MATRIX_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": json.dumps(user_payload, ensure_ascii=False),
                    },
                ],
            )
            content = (resp.choices[0].message.content or "").strip()
            last_raw = content[:200]
            if not content:
                last_err = "empty response content"
                continue
            payload = json.loads(content)
            if not validate_matrix_payload(payload):
                last_err = "schema validation failed"
                continue
            # ─── 모델이 안전책으로 IDENTITY 만 뱉었으면 거부하고 재시도. ──
            # 시스템 프롬프트가 명시적으로 금지하지만 gpt-4o-mini 가 종종
            # 무시한다. validate 단계에서 한 번 더 거른다.
            if _is_exact_identity(payload):
                last_err = "model returned exact identity (rejected)"
                last_raw = content[:200]
                continue
            _record_openai_diag(
                stage="ok",
                ok=True,
                detail=(
                    f"매트릭스 수신 OK (시도 {attempt + 1}/{max_retries + 1})"
                ),
                raw_preview=last_raw,
                model="gpt-4o-mini",
            )
            return payload
        except json.JSONDecodeError as e:
            last_err = f"json decode: {e}"
            continue
        except Exception as e:
            # 네트워크 / 레이트리밋 / 인증 등 — 예외 타입까지 보존
            last_err = f"{type(e).__name__}: {e}"
            continue

    # ─── 모든 시도가 실패하거나 IDENTITY 만 받았다면 결정론적 폴백 ────────
    # IDENTITY 를 그대로 반환하면 결과 이미지가 원본과 100% 동일해 사용자가
    # "AI 가 정말 호출됐나" 를 판단할 수 없다. illuminant 이름 기반 합리적
    # 행렬을 반환해 시각적 단서를 남긴다.
    illu_name = (rag_context.get("illuminant") or {}).get("name")
    fallback = _illuminant_fallback(illu_name)
    _record_openai_diag(
        stage="failed",
        ok=False,
        detail=(
            f"모든 시도 실패 (총 {max_retries + 1}회). 마지막 에러: {last_err}. "
            f"illuminant='{illu_name}' 기반 결정론적 폴백 사용."
        ),
        raw_preview=last_raw,
        model="gpt-4o-mini",
    )
    return fallback


# Backward-compat alias — 기존에 이미지 인자를 받던 시그니처가 코드 어딘가에 박혀있을 수
# 있으므로 keyword-only 로 image_bytes 를 받지만 무시한다.
def call_gpt4o_for_matrix(rag_context: dict, image_bytes: bytes | None = None) -> dict:
    """Deprecated. 이미지 입력은 무시되며 텍스트 전용 mini 호출로 위임된다."""
    return call_gpt4o_mini_for_matrix(rag_context)


# =============================================================================
# PHASE 3: LOCAL PIXEL TRANSFORM  —  numpy + OKLCH 5x10 명채도 타일
#   - numpy 로 3x4 행렬을 픽셀 전체에 실시간 적용 (마스킹은 향후 옵션)
#   - K-Means 등으로 대표 5색 추출 → palettes.base_colors 로만 저장
#   - 5x10 타일은 OKLCH 색공간에서 L 만 변주하여 클라이언트에서 즉시 재계산
# =============================================================================


# ---------- 단일 행렬 변환 (멀티리전 합성 루프가 반복 호출) -----------------------
IDENTITY: dict = {"r": [1, 0, 0, 0], "g": [0, 1, 0, 0], "b": [0, 0, 1, 0]}


def load_rgb(image_bytes: bytes) -> "np.ndarray":
    """PNG/JPEG bytes → H×W×3 float32 [0,1]."""
    if Image is None or np is None:
        raise NotImplementedError("Pillow/numpy 미설치")
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.asarray(img, dtype=np.float32) / 255.0


def to_png_bytes(arr: "np.ndarray") -> bytes:
    """H×W×3 float [0,1] → PNG bytes."""
    if Image is None or np is None:
        raise NotImplementedError("Pillow/numpy 미설치")
    arr_u8 = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr_u8).save(buf, "PNG")
    return buf.getvalue()


def apply_matrix_array(img: "np.ndarray", matrix: dict) -> "np.ndarray":
    """3×4 색변환 행렬을 H×W×3 float [0,1] 배열에 적용 (배열 in/out, 단일 행렬).

    멀티리전 합성 루프가 영역마다 반복 호출하는 핵심 단위 함수.
    `[r,g,b,1] @ M.T` 의 affine 형태로 bias(4번째 컬럼) 포함.
    """
    if np is None:
        raise NotImplementedError("numpy 미설치")
    M = np.array([matrix["r"], matrix["g"], matrix["b"]], dtype=np.float32)  # (3,4)
    H, W = img.shape[:2]
    rgb1 = np.concatenate(
        [img, np.ones((H, W, 1), dtype=np.float32)], axis=-1
    )  # (H,W,4)
    out = rgb1 @ M.T  # (H,W,3)
    return np.clip(out, 0.0, 1.0)


def apply_matrix(image_bytes: bytes, matrix: dict) -> bytes:
    """단일 영역(전체) 변환 — 바이트 in/out. `apply_matrix_array` 의 래퍼."""
    img = load_rgb(image_bytes)
    out = apply_matrix_array(img, matrix)
    return to_png_bytes(out)


def extract_base_colors(image_bytes: bytes, k: int = 5) -> list[str]:
    """결과 이미지에서 K-Means 로 대표 K개 HEX 색상 추출.

    sklearn 이 설치돼 있으면 KMeans, 없으면 평균 색 ×k 로 graceful degrade.
    """
    img = load_rgb(image_bytes)
    pixels = img.reshape(-1, 3)
    # 너무 큰 이미지는 다운샘플 (속도)
    if pixels.shape[0] > 50000:
        idx = np.random.default_rng(0).choice(pixels.shape[0], size=50000, replace=False)
        pixels = pixels[idx]
    try:
        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=k, n_init=4, random_state=0).fit(pixels)
        centers = km.cluster_centers_
    except Exception:
        centers = np.tile(pixels.mean(axis=0, keepdims=True), (k, 1))

    def _to_hex(c):
        r, g, b = (np.clip(c, 0, 1) * 255).astype(int).tolist()
        return f"#{r:02X}{g:02X}{b:02X}"

    return [_to_hex(c) for c in centers]


# ---------- OKLCH 기반 5×10 명채도 타일 -----------------------------------------
def make_shades(
    hex_color: str,
    n: int = 10,
    l_min: float = 0.15,
    l_max: float = 0.95,
) -> list[str]:
    """OKLCH 색공간에서 명도(L) 만 균등 변주하여 n 단계 색상 HEX 리스트를 생성.

    `coloraide` 의 `.fit()` 로 sRGB 게이먯 클램핑 → 양 끝 뭉개짐 방지.
    """
    try:
        from coloraide import Color
    except Exception:
        # coloraide 미설치 시 단순 fallback (원본만 n회 반복) — 사용자 가시성 위해 노출
        return [hex_color] * n

    base = Color(hex_color).convert("oklch")
    shades: list[str] = []
    for i in range(n):
        new_l = l_min + (l_max - l_min) * i / (n - 1)
        c = Color("oklch", [new_l, base["chroma"], base["hue"]])
        shades.append(c.convert("srgb").fit().to_string(hex=True))
    return shades


def build_5x10_tile_grid(base_colors: list[str]) -> list[list[str]]:
    """5색 × 10단계 명도 매트릭스(5x10)를 OKLCH 변주로 생성.

    이 결과는 **DB 에 저장하지 않는** 휘발성 데이터다 (설계 원칙).
    매번 `base_colors` 로부터 재계산되며 `st.session_state[SS_TILE_GRID]` 에만 캐싱된다.
    """
    if not base_colors:
        return []
    return [make_shades(c, n=10) for c in base_colors]


# =============================================================================
# PHASE 4: PERSISTENCE & STORAGE
#   - Supabase Auth 세션 체크
#   - Storage 버킷: /originals/ (원본) , /results/ (변환 결과) 분리
#   - DB INSERT 트랜잭션: projects → palettes (FK: palettes.project_id)
#     * 비즈니스 규칙: palettes 는 projects 보다 먼저 만들 수 없다.
#       projects ON DELETE CASCADE 로 palettes 가 자동 삭제됨.
#     * parent_id (projects 셀프 FK) 는 수정 스냅샷 / 스크랩 복제 시 채워짐.
#     * 신규 컬럼:
#         - subject_id  (FK -> reflectance_library)
#         - is_palette_public (BOOLEAN DEFAULT TRUE)
# =============================================================================

BUCKET_ORIGINALS = "originals"
BUCKET_RESULTS = "results"


def ensure_authenticated() -> dict | None:
    """Supabase Auth 세션 체크 — `sign_in_user` 가 채워준 `SS_AUTH_USER` 를 반환."""
    return st.session_state.get(SS_AUTH_USER)


def _resolve_auth():
    """`st_supabase_connection` 내부의 supabase-py Auth 클라이언트 추출.

    `_resolve_storage` 와 동일한 방어적 탐색 패턴.
    """
    for attr in ("client", "_client", "session", "supabase_client"):
        candidate = getattr(conn, attr, None)
        auth = getattr(candidate, "auth", None) if candidate is not None else None
        if auth is not None:
            return auth
    auth = getattr(conn, "auth", None)
    if auth is None:
        raise RuntimeError(
            "Supabase Auth 클라이언트를 찾을 수 없습니다. "
            "st_supabase_connection 버전을 확인하세요."
        )
    return auth


def _supabase_url_suffix(n: int = 18) -> str:
    """진단 메시지에 노출할 Supabase 프로젝트 URL 의 짧은 식별자.

    예: `https://abcdefghijkl.supabase.co` → `...lmnop.supabase.co`
    잘못된 프로젝트에 가입되고 있는지 사용자가 빠르게 검증 가능하도록 한다.
    전체 URL 은 노출하지 않는다 (시연 영상에 노출되더라도 안전).
    """
    try:
        url = s_url or ""
        return url[-n:] if url else "?"
    except Exception:
        return "?"


# ── profiles 셋업 SQL — 운영자가 Supabase SQL Editor 에 1회만 실행 ─────────────
# strict RLS (자기 행만 INSERT/UPDATE 가능) + SECURITY DEFINER 트리거 조합.
# 트리거가 가입 시점에 RLS 를 우회해 profiles 행을 즉시 만들어 주므로
# email-confirmation ON 환경(session=None) 에서도 가입이 막히지 않는다.
_PROFILES_SETUP_SQL = """ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can insert own profile" ON public.profiles
  FOR INSERT WITH CHECK (auth.uid() = id);
CREATE POLICY "Profiles are viewable by everyone" ON public.profiles
  FOR SELECT USING (true);
CREATE POLICY "Users can update own profile" ON public.profiles
  FOR UPDATE USING (auth.uid() = id);

CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER LANGUAGE plpgsql SECURITY DEFINER
SET search_path = public AS $$
BEGIN
  INSERT INTO public.profiles (id, nickname)
  VALUES (
    NEW.id,
    COALESCE(NEW.raw_user_meta_data->>'nickname',
             'user_' || substr(NEW.id::text, 1, 8))
  )
  ON CONFLICT (id) DO NOTHING;
  RETURN NEW;
END;
$$;

CREATE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();"""


def _is_rls_error(err: Exception | str) -> bool:
    """Postgres 42501 (insufficient_privilege) / RLS 위반 메시지인지 판별."""
    s = str(err)
    return (
        "42501" in s
        or "row-level security" in s.lower()
        or "violates row-level security policy" in s.lower()
    )


def _rls_guidance(table: str) -> str:
    """RLS 차단 시 사용자에게 어디에 무슨 SQL 을 붙여넣을지 안내.

    이 메시지가 보이면 운영자가 Supabase SQL Editor 에 본 프로젝트의 표준
    profiles 셋업 SQL (strict RLS + SECURITY DEFINER 트리거) 을 아직 적용하지
    않았다는 신호다. 적용 후엔 가입 시 트리거가 profiles 행을 자동 생성해
    이 분기 자체가 호출되지 않는다.
    """
    return (
        f"🔒 Postgres RLS 가 `public.{table}` 의 쓰기를 막고 있습니다.\n"
        f"Supabase Dashboard → SQL Editor 에 아래를 1회 실행해 주세요"
        f"(strict RLS + 자동 생성 트리거):\n\n"
        f"```sql\n{_PROFILES_SETUP_SQL}\n```"
    )


def sign_up_user(email: str, password: str, nickname: str) -> str:
    """이메일 + 비밀번호 회원가입.

    [본 프로젝트의 가입 흐름 — strict RLS + 자동 생성 트리거]
      * `auth.sign_up({email, password, options.data.nickname})` 를 호출하면
        Supabase 가 `auth.users` 에 행을 만들고, `raw_user_meta_data` 에
        `{"nickname": ...}` 을 저장한다.
      * `auth.users` INSERT 직후 트리거 `handle_new_user` (SECURITY DEFINER)
        가 자동으로 `public.profiles(id, nickname)` 을 INSERT 한다. RLS 가
        strict (`auth.uid() = id`) 여도 트리거가 시스템 권한으로 우회한다.
      * 따라서 클라이언트에서 별도의 `profiles.upsert` 가 필요 없다 — 트리거가
        단일 책임을 갖는다. 닉네임 변경은 `update_user_nickname` 으로 추후
        독립적으로 처리.

    [중복 가입 처리]
      * supabase-py v2 는 이미 존재하는 이메일에 대해 예외를 던지지 않는다.
        응답은 받지만 `session=None`, `identities==[]` 시그니처. 이걸로
        'duplicate' vs 'fresh signup' 을 구분한다.
    """
    if not email or not password:
        return "❌ 이메일과 비밀번호를 입력하세요."
    if len(password) < 6:
        return "❌ 비밀번호는 6자 이상이어야 합니다."

    url_suffix = _supabase_url_suffix()

    try:
        auth = _resolve_auth()
    except Exception as e:
        return f"❌ Auth 클라이언트 확보 실패: {e}"

    # 사용자가 빈 닉네임을 제출했어도 이메일 prefix 로 채워 트리거에 전달한다.
    # 트리거의 COALESCE 가 추가로 'user_<8>' fallback 을 가지지만, 이메일
    # prefix 가 식별성이 더 좋다.
    chosen_nick = (nickname or "").strip() or email.split("@")[0]

    try:
        res = auth.sign_up(
            {
                "email": email,
                "password": password,
                # raw_user_meta_data 에 들어가서 트리거 `handle_new_user` 가 읽음.
                # 이 값으로 `profiles.nickname` 이 자동 세팅된다.
                "options": {"data": {"nickname": chosen_nick}},
            }
        )
    except Exception as e:
        return (
            f"❌ Supabase sign_up 호출 실패\n"
            f"- 프로젝트: …{url_suffix}\n"
            f"- {type(e).__name__}: {e}"
        )

    user_obj = getattr(res, "user", None)
    user_id = getattr(user_obj, "id", None) if user_obj else None
    session = getattr(res, "session", None)
    identities = getattr(user_obj, "identities", None) if user_obj else None

    # supabase-py v2 의 'duplicate email' 시그니처
    if user_id and session is None and (identities == [] or identities is None):
        return (
            "⚠️ 이미 가입된 이메일입니다. 로그인 탭으로 진행하세요.\n"
            f"- 프로젝트: …{url_suffix}\n"
            "- 비밀번호를 잊었다면 Supabase Dashboard → Authentication → Users "
            "에서 비밀번호 재설정 메일을 발송하거나 사용자를 삭제 후 재가입하세요."
        )

    if not user_id:
        bits = []
        if hasattr(res, "session"):
            bits.append(f"session={bool(session)}")
        if hasattr(res, "user"):
            bits.append(f"user={bool(user_obj)}")
        return (
            f"⚠️ 가입 응답에 user.id 가 없습니다 ({', '.join(bits) or 'no diag'}).\n"
            f"- 프로젝트: …{url_suffix}\n"
            f"- Supabase Dashboard → Authentication → Settings 에서 "
            f"이메일 확인 정책을 확인하세요."
        )

    # auth.users INSERT 가 끝났으므로 트리거 `handle_new_user` 가 이미
    # profiles 행을 만들었어야 한다. 별도의 클라이언트 INSERT/UPSERT 는 하지
    # 않는다. (트리거 미적용 환경이라면 update_user_nickname 시점에 42501 이
    # 떨어지고 그때 _rls_guidance 가 본 PR 의 표준 SQL 을 안내해 준다.)
    return (
        f"✅ 가입 완료!\n"
        f"- user_id: `{user_id[:8]}…{user_id[-4:]}` (전체는 Supabase 에서 확인)\n"
        f"- 프로젝트: …{url_suffix}\n"
        f"- `auth.users` + `public.profiles` 양쪽 모두 생성됨\n"
        f"  (트리거 `handle_new_user` 가 profiles 행을 자동 생성)\n"
        f"- 닉네임: `{chosen_nick}` — 추후 '설정' 모달에서 변경 가능\n"
        f"- 확인 경로: **Supabase Dashboard → Authentication → Users**"
    )


def sign_in_user(email: str, password: str) -> str:
    """이메일 + 비밀번호 로그인 — `SS_AUTH_USER` 에 user 컨텍스트 저장."""
    if not email or not password:
        return "❌ 이메일과 비밀번호를 입력하세요."
    try:
        auth = _resolve_auth()
        res = auth.sign_in_with_password({"email": email, "password": password})
    except Exception as e:
        return f"❌ 로그인 실패: {e}"

    user_obj = getattr(res, "user", None)
    user_id = getattr(user_obj, "id", None) if user_obj else None
    if not user_id:
        return "❌ 로그인 실패: 사용자 정보를 받지 못했습니다."

    # 닉네임 조회 — profiles 에 없으면 이메일 prefix 로 fallback
    nickname: str | None = None
    try:
        prof = (
            conn.table("profiles")
            .select("nickname")
            .eq("id", user_id)
            .limit(1)
            .execute()
        )
        if prof.data:
            nickname = prof.data[0].get("nickname")
    except Exception:
        pass

    st.session_state[SS_AUTH_USER] = {
        "id": user_id,
        "email": getattr(user_obj, "email", email),
        "nickname": nickname or email.split("@")[0],
    }
    return "✅ 로그인 완료"


def sign_out_user() -> None:
    """Supabase Auth 세션 종료 + `SS_AUTH_USER` 초기화."""
    try:
        auth = _resolve_auth()
        auth.sign_out()
    except Exception:
        pass
    st.session_state.pop(SS_AUTH_USER, None)


def update_user_nickname(user_id: str, new_nickname: str) -> str:
    """`profiles.nickname` UPDATE 후 메시지 반환.

    Caller (`show_settings_modal`) 가 성공 시 `SS_AUTH_USER.nickname` 도
    같이 갱신해서 사이드바가 즉시 새 닉네임으로 보이게 한다.

    [에러 처리]
      * Postgres 42501 (RLS) → `_rls_guidance("profiles")` 로 사용자가 붙여넣을
        SQL 을 그대로 메시지에 첨부. 시연 중 막혀 있던 가장 흔한 케이스.
      * UPDATE 가 0건이면 UPSERT 로 폴백 (트리거가 없어 profiles 행이 아직
        없는 사용자도 닉네임을 설정할 수 있게).
    """
    cleaned = (new_nickname or "").strip()
    if not cleaned:
        return "❌ 닉네임은 비어 있을 수 없습니다."
    if len(cleaned) > 50:
        return "❌ 닉네임은 50자 이내로 입력하세요."

    try:
        resp = (
            conn.table("profiles")
            .update({"nickname": cleaned})
            .eq("id", user_id)
            .execute()
        )
    except Exception as e:
        if _is_rls_error(e):
            return "❌ 닉네임 변경 실패 (RLS 차단)\n\n" + _rls_guidance("profiles")
        return f"❌ 닉네임 변경 실패: {type(e).__name__}: {e}"

    # 일부 환경에서는 profiles 행이 아직 없어 UPDATE 가 0건일 수 있음 — UPSERT 폴백
    if not getattr(resp, "data", None):
        try:
            conn.table("profiles").upsert(
                {"id": user_id, "nickname": cleaned}
            ).execute()
        except Exception as e:
            if _is_rls_error(e):
                return (
                    "❌ profiles 행이 없고 UPSERT 도 RLS 가 막았습니다.\n\n"
                    + _rls_guidance("profiles")
                )
            return f"❌ profiles 행이 없어 UPSERT 도 실패: {type(e).__name__}: {e}"

    return f"✅ 닉네임이 '{cleaned}' 으로 변경되었습니다."


def _resolve_storage():
    """`st_supabase_connection` 내부의 supabase-py Storage 클라이언트를 안전하게 추출.

    버전에 따라 `conn.client` / `conn._client` / `conn.session` 중 노출 위치가
    다를 수 있으므로 방어적으로 탐색한다.
    """
    for attr in ("client", "_client", "session", "supabase_client"):
        candidate = getattr(conn, attr, None)
        storage = getattr(candidate, "storage", None) if candidate is not None else None
        if storage is not None:
            return storage
    # 마지막 시도: conn 자체가 storage 속성을 직접 노출하는 경우
    storage = getattr(conn, "storage", None)
    if storage is not None:
        return storage
    raise RuntimeError(
        "Supabase Storage 클라이언트를 찾을 수 없습니다. "
        "st_supabase_connection 버전을 확인하세요."
    )


def upload_to_storage(
    bucket: str,
    path: str,
    data: bytes,
    *,
    content_type: str = "image/png",
) -> str:
    """Supabase Storage 업로드 후 public URL 반환.

    경로 스킴: 호출자가 `f"{uuid}/origin.png"` 형태로 unique key 를 만들어 넘긴다.
    같은 키로 두 번 호출돼도 `upsert=true` 로 덮어쓴다 (수정/재시도 안전).

    Args:
        bucket: Supabase Storage 버킷 이름 (예: `originals`, `results`).
        path:   버킷 내부 경로 (UUID 기반 유니크 키).
        data:   업로드할 바이너리.
        content_type: MIME — 기본 image/png. JPEG 등으로 호출 가능.

    Returns:
        public URL 문자열. (private 버킷이라면 호출 측에서 signed URL 로 교체)
    """
    storage = _resolve_storage()
    bucket_api = storage.from_(bucket)
    # supabase-py v2: upload(path, file=..., file_options={...})
    try:
        bucket_api.upload(
            path=path,
            file=data,
            file_options={"content-type": content_type, "upsert": "true"},
        )
    except TypeError:
        # 구버전 supabase-py 호환 — positional 시그니처
        bucket_api.upload(path, data, {"content-type": content_type, "upsert": "true"})
    return bucket_api.get_public_url(path)


def save_simulation_result(
    user_id,
    matrix,
    origin_url,
    base_colors,
    *,
    subject_id: str | None = None,
    illu_id: str | None = None,
    result_url: str | None = None,
    is_anonymous: bool | None = None,
    is_palette_public: bool | None = None,
    subject_labels: list[str] | None = None,
):
    """프로젝트 + 팔레트 동시 저장 (기존 시그니처 보호 + 신규 키워드 인자).

    [데이터 무결성 규칙]
      * projects 가 먼저 INSERT 되어 `id` 가 확정되어야 palettes.project_id 가
        충족된다. 두 호출 사이에서 예외 발생 시 projects 만 남는 고아 레코드가
        생길 수 있으므로 향후 RPC(트랜잭션 함수) 로 승격할 것.
      * `parent_id` 셀프 FK 는 본 함수에서는 NULL 이며, 수정/스크랩 흐름에서
        `save_snapshot()` / `scrap_project()` 가 채워준다.
      * `palettes` 는 projects 에 대해 ON DELETE CASCADE 이므로 별도 정리 불필요.
      * `subject_id` = "대표 피사체" (멀티리전 시 첫 영역). 모든 영역 라벨은
        `subject_labels` JSONB 배열로 함께 저장된다.
      * `matrix_json` 은 멀티리전의 경우 재현 불가(마스크 미저장)이므로
        디버그/감사 용도로만 유지한다.
    """
    data_to_insert = {
        "owner_id": user_id,
        "subject_id": subject_id,
        "subject_labels": subject_labels,
        "illu_id": illu_id,
        "matrix_json": matrix,
        "origin_path": origin_url,
        "result_path": result_url,
        "is_anonymous": (
            is_anonymous
            if is_anonymous is not None
            else st.session_state.get(SS_IS_ANON, False)
        ),
        "is_palette_public": (
            is_palette_public
            if is_palette_public is not None
            else st.session_state.get(SS_IS_PALETTE_PUBLIC, True)
        ),
        "status": "Active",
        "scrap_count": 0,
    }
    # None 값은 DB default 로 위임 (NOT NULL 컬럼 보호)
    data_to_insert = {k: v for k, v in data_to_insert.items() if v is not None}

    project_resp = conn.table("projects").insert(data_to_insert).execute()

    if project_resp.data:
        try:
            row = (
                project_resp.data[0]
                if isinstance(project_resp.data, list)
                else project_resp.data
            )
            p_id = row["id"]
            conn.table("palettes").insert(
                {"project_id": p_id, "base_colors": base_colors}
            ).execute()
            return f"✅ DB 저장 성공! (ID: {p_id})"
        except Exception:
            return "✅ DB 저장은 성공했으나, 화면 표시 중 작은 오류가 발생했습니다. (DB 확인 요망)"
    return "❌ 저장 실패"


def save_snapshot(parent_project_id: str, matrix, origin_url, result_url, base_colors) -> str | None:
    """기존 프로젝트의 수정본을 새 row 로 INSERT (스냅샷).

    [의도적으로 Phase 2 범위 외]
      * Phase 2 멀티리전 사양은 **마스크를 DB에 저장하지 않는다**. 따라서 합성
        결과를 정확히 재현할 수 없어 "원본을 편집해 새 버전을 만든다" 라는
        스냅샷 의미가 약해진다 (cursor 작업 지시: "재현/편집 안 할 거라 과한
        정규화는 불필요").
      * `scrap_project` 가 익명 카운트만 +1 하는 것으로 Phase 2 의 모든
        '재공유' UX 를 흡수한다. 따라서 본 함수는 현재 호출되지 않는다.
      * 향후 마스크 직렬화 정책이 결정되면 본 함수를 살려서:
          - parent_id = parent_project_id 로 새 projects row INSERT
          - 동일 palettes row 복제 (또는 base_colors 만 새로 저장)
          - 그래프 형태 히스토리(`parent_id` 셀프 FK) 유지
        형태로 구현한다.
    """
    raise NotImplementedError(
        "save_snapshot 은 Phase 2 범위 외 — 마스크 미저장 정책상 재현 불가 "
        "(필요해질 때 활성화)."
    )


# =============================================================================
# PHASE 5: BUSINESS LOGIC  —  갤러리 / 스크랩(익명 카운트) / 조건부 삭제
#   - '익명의 창작자' DB View 호출 (예: anon_gallery_view)
#   - 스크랩은 익명 카운트만 누적: scrap_count += 1 (누가 했는지는 추적하지 않음)
#   - 조건부 삭제: scrap_count >= 1 OR (now - created_at) < 30일  →
#       즉시 삭제 불가, status = 'Pending_Deletion' 로 전이 + 배너 노출
# =============================================================================

SHARE_LOCK_DAYS = 30  # 공유 후 즉시 삭제 잠금 기간


_PROJECT_SELECT = (
    "id, owner_id, origin_path, result_path, scrap_count, created_at, status, "
    "subject_id, subject_labels, illu_id, is_anonymous, is_palette_public"
)


def fetch_my_shared_projects(owner_id: str | None, limit: int = 20) -> list[dict]:
    """좌측 사이드바 — '내가 공유한' 라인업."""
    if not owner_id:
        return []
    try:
        resp = (
            conn.table("projects")
            .select(_PROJECT_SELECT)
            .eq("owner_id", owner_id)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return resp.data or []
    except Exception:
        return []


def fetch_others_shared_projects(owner_id: str | None, limit: int = 20) -> list[dict]:
    """좌측 사이드바 — '남이 공유한' 라인업 (anon_gallery_view 우선, 실패 시 projects fallback)."""
    try:
        q = conn.table("anon_gallery_view").select("*").limit(limit)
        if owner_id:
            q = q.neq("owner_id", owner_id)
        resp = q.execute()
        return resp.data or []
    except Exception:
        try:
            q = (
                conn.table("projects")
                .select(_PROJECT_SELECT)
                .eq("is_anonymous", True)
                .order("created_at", desc=True)
                .limit(limit)
            )
            if owner_id:
                q = q.neq("owner_id", owner_id)
            resp = q.execute()
            return resp.data or []
        except Exception:
            return []


def fetch_palette_for_project(project_id: str) -> list[str]:
    """프로젝트에 매달린 palettes.base_colors 조회."""
    try:
        resp = (
            conn.table("palettes")
            .select("base_colors")
            .eq("project_id", project_id)
            .limit(1)
            .execute()
        )
        data = resp.data or []
        if not data:
            return []
        return list(data[0].get("base_colors") or [])
    except Exception:
        return []


def scrap_project(source_project_id: str, current_scrap_count: int) -> str:
    """타인의 프로젝트를 스크랩 — 익명 카운트만 +1.

    [비즈니스 규칙]
      * 누가 스크랩했는지는 추적하지 않는다 (요구사항: "인원수만").
      * 원본 status 가 'Pending_Deletion' 이면 스크랩 불가 (UI 단에서 버튼 비활성).
    """
    try:
        conn.table("projects").update(
            {"scrap_count": (current_scrap_count or 0) + 1}
        ).eq("id", source_project_id).execute()
        return "✅ 스크랩 완료 (익명 카운트 +1)"
    except Exception as e:
        return f"❌ 스크랩 실패: {e}"


def request_project_deletion(project_id: str, created_at_iso: str, scrap_count: int) -> str:
    """삭제 요청 — 조건부 분기.

    * scrap_count >= 1   → 즉시 삭제 불가 (다른 사용자가 스크랩 사용 중)
    * 공유 후 30일 미만  → 즉시 삭제 불가
    * 둘 다 해당 없음    → 실제 DELETE (CASCADE 로 palettes 동반 삭제)
    """
    try:
        created_at = datetime.fromisoformat(created_at_iso.replace("Z", "+00:00"))
    except Exception:
        created_at = datetime.utcnow()

    locked_by_scrap = (scrap_count or 0) >= 1
    locked_by_period = (
        datetime.utcnow() - created_at.replace(tzinfo=None)
    ) < timedelta(days=SHARE_LOCK_DAYS)

    if locked_by_scrap or locked_by_period:
        try:
            conn.table("projects").update({"status": "Pending_Deletion"}).eq(
                "id", project_id
            ).execute()
        except Exception:
            pass
        reasons = []
        if locked_by_scrap:
            reasons.append(f"스크랩 {scrap_count}건")
        if locked_by_period:
            reasons.append(f"공유 후 {SHARE_LOCK_DAYS}일 미경과")
        return f"⏳ 삭제 대기 상태로 전환되었습니다 ({', '.join(reasons)})."

    # TODO[PHASE-5]: 즉시 DELETE — conn.table("projects").delete().eq("id", project_id).execute()
    return "🗑️ 즉시 삭제 가능 (실제 DELETE 호출은 PHASE 5 구현 시 활성화)."


# =============================================================================
# UI COMPONENTS
# =============================================================================

# ---------- 좌측 사이드바: 인증 + 라인업 ----------------------------------------
def render_sidebar() -> None:
    """좌측: 인증 패널 + '내가 공유한' / '남이 공유한' 미리보기 라인업.

    프로필 이미지는 보류(요구사항). 닉네임 + UID 만 노출.
    """
    with st.sidebar:
        _render_auth_panel()
        st.divider()

        user = ensure_authenticated()
        if user:
            st.caption("내가 공유한 미리보기")
            _render_lineup(
                fetch_my_shared_projects(user.get("id"), limit=10),
                key_prefix="mine",
            )
            st.divider()

        st.caption("내가 공유받은(남이 공유한) 미리보기")
        _render_lineup(
            fetch_others_shared_projects((user or {}).get("id"), limit=10),
            key_prefix="others",
        )


def _render_auth_panel() -> None:
    """사이드바 상단 — 로그인/회원가입 (비인증) 또는 프로필 + 설정 + 로그아웃 (인증).

    아이콘(⚙️ / ⎋) 만 사용했을 때 일부 환경의 emoji 폰트가 fallback 정사각형으로
    렌더되어 가독성이 떨어졌다. 텍스트 라벨 + 단순 ASCII 마커를 함께 사용해
    OS/브라우저 무관하게 안정적으로 보이게 한다.
    """
    user = ensure_authenticated()
    if user:
        # 1행: 닉네임 + UID
        st.markdown(f"**{user.get('nickname') or user.get('email') or 'user'}**")
        uid_short = (user.get("id") or "")[:8]
        st.caption(f"@{uid_short or 'user_UID'}")

        # 2행: 두 개의 동일 너비 텍스트 버튼 — 어떤 환경에서도 깔끔하게 보임
        col_a, col_b = st.columns(2)
        if col_a.button(
            "설정",
            key="settings_btn",
            help="닉네임 변경 / DB 진단",
            **_FW,
        ):
            show_settings_modal()
        if col_b.button(
            "로그아웃",
            key="signout_btn",
            help="현재 세션 종료",
            **_FW,
        ):
            sign_out_user()
            st.rerun()
        return

    # ── 비로그인 상태 — 인라인 폼 ──────────────────────────────────────────
    st.markdown("**🔐 로그인 / 회원가입**")
    tab_in, tab_up = st.tabs(["로그인", "회원가입"])

    with tab_in:
        with st.form("signin_form", clear_on_submit=False):
            in_email = st.text_input("이메일", key="signin_email")
            in_pwd = st.text_input("비밀번호", type="password", key="signin_pwd")
            submitted_in = st.form_submit_button(
                "로그인", **_FW, type="primary"
            )
        if submitted_in:
            msg = sign_in_user(in_email, in_pwd)
            if msg.startswith("✅"):
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)

    with tab_up:
        with st.form("signup_form", clear_on_submit=False):
            up_email = st.text_input("이메일", key="signup_email")
            up_pwd = st.text_input(
                "비밀번호 (6자 이상)", type="password", key="signup_pwd"
            )
            up_nick = st.text_input("닉네임 (선택)", key="signup_nick")
            submitted_up = st.form_submit_button(
                "가입", **_FW
            )
        if submitted_up:
            msg = sign_up_user(up_email, up_pwd, up_nick)
            # 다중 라인 + 마크다운 가독성 보장: success/warning/error 헬퍼는
            # 마크다운을 지원하지만 narrow sidebar 에서는 info 박스 모양이
            # 더 안정적으로 줄바꿈된다.
            if msg.startswith("✅"):
                st.success(msg)
            elif msg.startswith("⚠️"):
                st.warning(msg)
            else:
                st.error(msg)


def _render_lineup(items: list[dict], key_prefix: str) -> None:
    """라인업 카드 (작은 미리보기 + 클릭 시 모달 오픈)."""
    if not items:
        st.caption("_(아직 공유된 항목이 없습니다)_")
        return
    cols = st.columns(2)
    for i, proj in enumerate(items):
        with cols[i % 2]:
            with st.container(border=True):
                st.markdown(
                    f"<div style='height:60px;background:#f0f0f0;border-radius:4px;"
                    f"display:flex;align-items:center;justify-content:center;font-size:11px;color:#888;'>"
                    f"image prev</div>",
                    unsafe_allow_html=True,
                )
                # 팔레트 5색 strip
                colors = fetch_palette_for_project(proj["id"])[:5]
                if colors:
                    strip = "".join(
                        f"<div style='flex:1;height:14px;background:{c};'></div>" for c in colors
                    )
                    st.markdown(
                        f"<div style='display:flex;margin-top:4px;border-radius:4px;overflow:hidden;'>{strip}</div>",
                        unsafe_allow_html=True,
                    )
                st.caption(f"palette {len(colors)}color")
                if st.button(
                    "Palette Title",
                    key=f"{key_prefix}_open_{proj['id']}",
                    **_FW,
                ):
                    show_project_modal(proj, mode="view")


# ---------- 중앙: 원본 이미지 + 태그 패널 ---------------------------------------
def render_original_image_panel() -> None:
    """중앙 상단: 원본 이미지 프리뷰."""
    with st.container(border=True):
        img = st.session_state.get(SS_ORIGIN_IMAGE)
        if img is not None:
            st.image(img, **_FW)
        else:
            st.markdown(
                "<div style='height:320px;display:flex;align-items:center;"
                "justify-content:center;font-size:28px;color:#888;'>"
                "Original<br>Image<br>Preview</div>",
                unsafe_allow_html=True,
            )


def render_tag_panel() -> None:
    """중앙 하단: 피사체(영역별 N개) / 광원(단일) 태그 탭."""
    st.markdown("**Tags**")
    tabs = st.tabs(["🦆 피사체 영역 (Subject · N개)", "💡 광원 (Illuminant · 5)"])
    with tabs[0]:
        # [버그픽스] 피사체 패널에서 예외가 나도 광원 탭/하위 UI 가 통째로
        # 사라지지 않도록 격리한다. (과거: 영역 추가 시 캔버스 렌더 실패가
        # 스크립트 전체를 중단시켜 광원 선택이 불가능해지던 증상)
        try:
            _render_region_panel()
        except Exception as e:  # pragma: no cover - defensive
            st.error(f"피사체 영역 패널 오류: {type(e).__name__}: {e}")
    with tabs[1]:
        try:
            _render_illuminant_tag_grid()
        except Exception as e:  # pragma: no cover - defensive
            st.error(f"광원 패널 오류: {type(e).__name__}: {e}")
    st.caption("Orange: Selected · Gray: Not Selected · 같은 광원 아래 여러 영역에 다른 피사체를 매핑합니다.")


def _render_region_panel() -> None:
    """영역(region) 리스트 관리 + 새 영역 추가 위젯.

    영역 = {subject, mask, matrix}. 마스크는 휘발(DB 미저장).
    """
    regions: list[dict] = st.session_state.setdefault(SS_REGIONS, [])

    # ─── 현재 영역 목록 ───
    if regions:
        st.markdown(f"**현재 영역: {len(regions)}개**")
        for i, region in enumerate(list(regions)):
            subj = region.get("subject") or {}
            label = subj.get("label", "(unknown)")
            mask = region.get("mask")
            mask_info = (
                f"{int(mask.sum())} px 칠해짐"
                if mask is not None and np is not None
                else "마스크 없음 → 전체 적용"
            )
            with st.expander(
                f"#{i + 1} · {label}  ·  {mask_info}",
                expanded=(i == len(regions) - 1),
            ):
                # ── 변환 스타일: 무지개(구조색) vs 단색 ──
                style = st.radio(
                    "변환 스타일",
                    options=["무지개 (구조색 광택)", "단색 (광원 색조)"],
                    index=0 if region.get("irid_mode", "iridescent") == "iridescent" else 1,
                    key=f"irid_mode_radio_{i}",
                    horizontal=True,
                    help=(
                        "무지개: 까마귀 깃털처럼 부위별로 파랑/보라/청록이 다르게 "
                        "나타나는 영롱한 효과. 단색: 광원 색조를 평평하게 입힘."
                    ),
                )
                region["irid_mode"] = (
                    "iridescent" if style.startswith("무지개") else "flat"
                )
                if region["irid_mode"] == "iridescent":
                    region["irid_boost"] = st.slider(
                        "무지개 강도",
                        min_value=2.0,
                        max_value=14.0,
                        value=float(region.get("irid_boost", 7.0)),
                        step=0.5,
                        key=f"irid_boost_{i}",
                        help="클수록 색이 더 강하게 펼쳐집니다.",
                    )

                _render_mask_canvas_for_region(i, region)
                if st.button("🗑️ 이 영역 삭제", key=f"del_region_{i}"):
                    regions.pop(i)
                    st.session_state[SS_REGIONS] = regions
                    st.rerun()

        st.divider()

    # ─── 새 영역 추가 ───
    st.markdown("**+ 새 영역의 피사체 선택**")
    q = st.text_input(
        "Tag Search",
        key="subject_search",
        placeholder="검색 (예: Mallard, Bird, Flower …)",
        label_visibility="collapsed",
    )
    subjects = list_subjects()
    if q:
        ql = q.lower()
        subjects = [
            s for s in subjects
            if ql in (s.get("label") or "").lower()
            or ql in (s.get("category") or "").lower()
        ]

    if not subjects:
        st.caption("_(피사체 데이터가 없습니다. reflectance_library 에 INSERT 가 필요합니다)_")
        return

    used_ids = {(r.get("subject") or {}).get("id") for r in regions}

    cols = st.columns(4)
    for i, subj in enumerate(subjects):
        with cols[i % 4]:
            already_used = subj["id"] in used_ids
            label = subj["label"]
            if subj.get("is_uv_active"):
                label = f"✦ {label}"
            if already_used:
                label = f"✓ {label}"
            if st.button(
                label,
                key=f"add_region_{subj['id']}",
                type=("primary" if already_used else "secondary"),
                **_FW,
                help=("이미 영역으로 추가됨 — 다시 누르면 영역이 하나 더 생성됩니다." if already_used else None),
            ):
                regions.append({"subject": subj, "mask": None, "matrix": None})
                st.session_state[SS_REGIONS] = regions
                st.rerun()


def _render_mask_canvas_for_region(region_index: int, region: dict) -> None:
    """영역별 drawable-canvas 마스크 위젯.

    [좌표계 동기화 — 제일 흔한 버그]
      * canvas 표시 크기 ≠ 실제 이미지(H×W) 이면 마스크가 어긋난다.
      * 따라서 canvas 알파 채널을 `cv2.resize(..., (W, H), INTER_NEAREST)` 로 리사이즈.
      * **boolean 마스크에 선형 보간을 쓰면 경계가 회색으로 뭉개진다 → 반드시 INTER_NEAREST.**
    """
    img_pil = st.session_state.get(SS_ORIGIN_IMAGE)
    if img_pil is None:
        st.caption("먼저 좌측 'Upload' 로 이미지를 업로드하세요.")
        return
    if np is None:
        st.warning("numpy 가 필요합니다 (requirements 설치).")
        return

    # ─────────────────────────────────────────────────────────────────────────
    # [버그픽스] 기본 마스킹 경로 = bbox 슬라이더 (항상 안전).
    #
    # 과거: st_canvas 를 먼저 호출하고 예외 시 폴백 → 그러나 streamlit-drawable-
    # canvas 가 현재 Streamlit 버전과 비호환일 때, 호출이 '파이썬 예외' 가 아니라
    # 위젯 렌더링 단계에서 스크립트 실행을 중단시켜 try/except 로 못 잡고 그 아래
    # (광원 탭 / Convert 버튼 / 결과 패널) 전체가 사라지는 치명적 증상이 있었다.
    #
    # 해결: bbox 슬라이더를 기본으로 항상 렌더링하고, freedraw 캔버스는 사용자가
    # 체크박스로 명시적으로 켤 때만 시도한다. import 와 호출을 모두 가드해
    # 어떤 경우에도 패널 하단이 죽지 않도록 한다.
    # ─────────────────────────────────────────────────────────────────────────
    use_canvas = st.checkbox(
        "🖌️ 자유곡선(freedraw) 캔버스 사용 (실험적 · 환경에 따라 비활성)",
        value=False,
        key=f"use_canvas_{region_index}",
        help=(
            "체크 해제 시 사각 영역 슬라이더로 마스킹합니다(권장 · 항상 동작). "
            "캔버스는 streamlit-drawable-canvas 호환 환경에서만 동작합니다."
        ),
    )

    if not use_canvas:
        _render_bbox_mask_fallback(region_index, region, img_pil)
        return

    # ── 캔버스 옵트인 경로 ──
    if cv2 is None:
        st.warning("opencv-python-headless 가 필요합니다 → 슬라이더 마스킹으로 진행합니다.")
        _render_bbox_mask_fallback(region_index, region, img_pil)
        return
    try:
        from streamlit_drawable_canvas import st_canvas
    except Exception as exc:
        _render_bbox_mask_fallback(region_index, region, img_pil, canvas_error=exc)
        return

    iw, ih = img_pil.size  # PIL: (W, H)
    DISPLAY_W = 480
    scale = DISPLAY_W / iw
    disp_h = int(round(ih * scale))

    stroke = st.slider(
        "브러시 두께",
        min_value=4,
        max_value=80,
        value=24,
        key=f"stroke_{region_index}",
    )

    try:
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=stroke,
            stroke_color="#FFA500",
            background_image=img_pil,
            update_streamlit=True,
            width=DISPLAY_W,
            height=disp_h,
            drawing_mode="freedraw",
            key=f"canvas_region_{region_index}",
        )
    except Exception as exc:
        # 캔버스 호출이 예외로 끝나면 슬라이더 폴백으로 전환 (진단 expander 포함).
        _render_bbox_mask_fallback(region_index, region, img_pil, canvas_error=exc)
        return

    # canvas 알파 채널을 이진화 → 실제 이미지 좌표계로 INTER_NEAREST 리사이즈
    if canvas_result is None or canvas_result.image_data is None:
        return
    raw_alpha = canvas_result.image_data[:, :, 3]
    if raw_alpha.sum() == 0:
        # 아직 안 칠함 — 마스크 없음 (None == 전체 적용)
        region["mask"] = None
        return
    raw_mask = (raw_alpha > 0).astype(np.uint8)
    mask_native = cv2.resize(raw_mask, (iw, ih), interpolation=cv2.INTER_NEAREST)
    region["mask"] = mask_native.astype(bool)


def _grabcut_foreground(
    img_pil,
    px1: int,
    py1: int,
    px2: int,
    py2: int,
    iters: int = 5,
) -> "np.ndarray | None":
    """사각형(px1,py1,px2,py2) 안에서 GrabCut 으로 전경(피사체)만 추출.

    배경 제외의 핵심: 사용자가 사각형으로 새를 대충 감싸면, 그 안에서 OpenCV
    GrabCut 이 색/질감 통계로 전경(새)과 배경(물/얼음)을 자동 분리한다.
    반환: H×W bool 마스크 (전경 True) 또는 실패 시 None.
    """
    if cv2 is None or np is None:
        return None
    try:
        arr = np.asarray(img_pil.convert("RGB"))
        ih, iw = arr.shape[:2]
        # 사각형이 너무 작으면 의미 없음
        if (px2 - px1) < 8 or (py2 - py1) < 8:
            return None
        gc_mask = np.zeros((ih, iw), np.uint8)
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        rect = (px1, py1, px2 - px1, py2 - py1)
        # 속도: 큰 이미지는 다운스케일해서 GrabCut 후 업스케일
        scale = 1.0
        work = arr
        max_side = max(ih, iw)
        if max_side > 600:
            scale = 600.0 / max_side
            work = cv2.resize(arr, (int(iw * scale), int(ih * scale)),
                              interpolation=cv2.INTER_AREA)
            gc_mask = np.zeros(work.shape[:2], np.uint8)
            rect = (int(px1 * scale), int(py1 * scale),
                    int((px2 - px1) * scale), int((py2 - py1) * scale))
        cv2.grabCut(work, gc_mask, rect, bgd, fgd, iters, cv2.GC_INIT_WITH_RECT)
        # GC_FGD(1) / GC_PR_FGD(3) 를 전경으로
        fg = np.where((gc_mask == 1) | (gc_mask == 3), 1, 0).astype(np.uint8)
        if scale != 1.0:
            fg = cv2.resize(fg, (iw, ih), interpolation=cv2.INTER_NEAREST)
        if fg.sum() == 0:
            return None
        return fg.astype(bool)
    except Exception:
        return None


def _render_bbox_mask_fallback(
    region_index: int,
    region: dict,
    img_pil,
    *,
    canvas_error: Exception | None = None,
) -> None:
    """사각 영역 마스크 UI + GrabCut 전경 자동 추출.

    1) 4개 슬라이더로 새를 대충 감싸는 사각형 지정
    2) '윤곽 자동 추출(배경 제외)' 버튼 → GrabCut 으로 새만 분리
    region['mask'] 에 boolean 마스크를 저장한다.
    """
    if Image is None or np is None:
        st.warning("Pillow / numpy 가 필요합니다 (requirements 설치).")
        return

    iw, ih = img_pil.size  # PIL: (W, H)

    bcols = st.columns(2)
    with bcols[0]:
        x1 = st.slider(
            "좌측 (%)", 0, 99, 10, key=f"bbox_x1_{region_index}"
        )
        y1 = st.slider(
            "상단 (%)", 0, 99, 10, key=f"bbox_y1_{region_index}"
        )
    with bcols[1]:
        x2 = st.slider(
            "우측 (%)", 1, 100, 90, key=f"bbox_x2_{region_index}"
        )
        y2 = st.slider(
            "하단 (%)", 1, 100, 90, key=f"bbox_y2_{region_index}"
        )

    if x1 >= x2 or y1 >= y2:
        st.warning("범위가 올바르지 않습니다 (좌측 < 우측, 상단 < 하단).")
        region["mask"] = None
        return

    px1, px2 = int(iw * x1 / 100), int(iw * x2 / 100)
    py1, py2 = int(ih * y1 / 100), int(ih * y2 / 100)

    # ── GrabCut: 사각형 안에서 전경(새)만 자동 분리 → 배경 제외 ──
    gc_key = f"grabcut_mask_{region_index}"
    btn_cols = st.columns(2)
    if btn_cols[0].button(
        "✂️ 윤곽 자동 추출 (배경 제외)",
        key=f"grabcut_btn_{region_index}",
        help="사각형 안에서 피사체만 자동으로 따냅니다 (배경/물/얼음 제외).",
        disabled=(cv2 is None),
    ):
        with st.spinner("피사체 윤곽 추출 중…"):
            fg = _grabcut_foreground(img_pil, px1, py1, px2, py2)
        if fg is not None:
            st.session_state[gc_key] = fg
            st.toast("윤곽 추출 완료 — 배경이 제외됩니다.")
        else:
            st.session_state.pop(gc_key, None)
            st.warning("윤곽 추출 실패 — 사각형을 더 크게/명확하게 잡아보세요.")
    if btn_cols[1].button(
        "↩️ 사각형으로 되돌리기",
        key=f"grabcut_reset_{region_index}",
        help="자동 추출을 취소하고 사각형 전체를 마스크로 사용합니다.",
    ):
        st.session_state.pop(gc_key, None)

    gc_mask = st.session_state.get(gc_key)

    # 최종 마스크 결정: GrabCut 결과가 있으면 그걸, 없으면 사각형 전체
    if gc_mask is not None and gc_mask.shape == (ih, iw):
        mask = gc_mask
        mask_kind = "윤곽(배경 제외)"
    else:
        mask = np.zeros((ih, iw), dtype=bool)
        mask[py1:py2, px1:px2] = True
        mask_kind = "사각형"
    region["mask"] = mask

    # 시각 피드백 — 마스크 영역에 반투명 오렌지 오버레이
    arr = np.asarray(img_pil.convert("RGB")).copy().astype(np.float32)
    overlay_color = np.array([255, 165, 0], dtype=np.float32)
    sel = mask
    arr[sel] = np.clip(arr[sel] * 0.55 + overlay_color * 0.45, 0, 255)
    st.image(Image.fromarray(arr.astype(np.uint8)), **_FW)
    st.caption(
        f"마스크: {mask_kind} · {int(mask.sum())} 픽셀 "
        f"({int(mask.sum()) * 100 // (iw * ih)}% 영역)"
    )

    # 진단 (canvas_error 있을 때만, expander 중첩 금지 → container)
    if canvas_error is not None:
        with st.container(border=True):
            st.caption("🔧 캔버스 진단 (왜 슬라이더로 폴백?)")
            st.code(f"streamlit: {st.__version__}", language="text")
            st.code(
                f"canvas 호출 예외: {type(canvas_error).__name__}: {canvas_error}",
                language="text",
            )
    return


def _render_illuminant_tag_grid() -> None:
    illus = list_illuminants()
    sel = st.session_state.get(SS_SELECTED_ILLU) or {}
    sel_id = sel.get("id")

    if not illus:
        st.caption("_(광원 데이터가 없습니다. standard_illuminants 에 INSERT 가 필요합니다)_")
        return

    cols = st.columns(min(5, len(illus)))
    for i, illu in enumerate(illus):
        with cols[i % len(cols)]:
            is_sel = illu["id"] == sel_id
            if st.button(
                illu["name"],
                key=f"illu_{illu['id']}",
                type=("primary" if is_sel else "secondary"),
                **_FW,
                help=illu.get("description") or "",
            ):
                if is_sel:
                    st.session_state[SS_SELECTED_ILLU] = None
                else:
                    st.session_state[SS_SELECTED_ILLU] = illu
                st.rerun()


# ---------- 우측: 변환 결과 + 5x10 그리드 ---------------------------------------
def render_converted_image_panel() -> None:
    with st.container(border=True):
        result = st.session_state.get(SS_RESULT_BYTES)
        if result:
            st.image(result, **_FW)
        else:
            st.markdown(
                "<div style='height:240px;display:flex;align-items:center;"
                "justify-content:center;font-size:24px;color:#888;'>"
                "Converted<br>Image<br>Preview</div>",
                unsafe_allow_html=True,
            )


def render_5x10_grid() -> None:
    """5×10 명채도 매트릭스 (휘발성 — DB 미저장)."""
    with st.container(border=True):
        grid = st.session_state.get(SS_TILE_GRID) or []
        if not grid:
            st.markdown(
                "<div style='height:200px;display:flex;align-items:center;"
                "justify-content:center;font-size:28px;color:#888;'>5 × 10</div>",
                unsafe_allow_html=True,
            )
            return
        st.caption("5 × 10  (OKLCH 명도 변주 · 휘발성)")
        for row in grid:
            row_html = "".join(
                f"<div style='flex:1;height:24px;background:{hex_code};'></div>"
                for hex_code in row
            )
            st.markdown(
                f"<div style='display:flex;gap:1px;margin-bottom:1px;border-radius:3px;overflow:hidden;'>{row_html}</div>",
                unsafe_allow_html=True,
            )


# ---------- 중앙↔우측 사이의 세로 버튼 컬럼 -------------------------------------
def _reset_workspace(keep_login: bool = True) -> None:
    """작업 상태(이미지/영역/광원/결과)를 비운다. 로그인 세션은 유지.

    이미지나 태그를 바꾸려 할 때 새로고침+재로그인 없이 깨끗이 초기화한다.
    """
    clear_keys = [
        SS_ORIGIN_IMAGE, SS_ORIGIN_BYTES, SS_RESULT_BYTES, SS_MATRIX,
        SS_BASE_COLORS, SS_TILE_GRID, SS_SELECTED_ILLU, SS_REGIONS,
        "_last_upload_name",
    ]
    for k in clear_keys:
        st.session_state.pop(k, None)

    # 위젯 상태(업로더/캔버스/슬라이더/검색 등)도 정리 — 잔여 키가 남으면
    # 새 이미지에 옛 마스크/슬라이더가 따라붙는다. 접두사로 일괄 제거.
    prefixes = (
        "uploader", "canvas_region_", "use_canvas_", "stroke_",
        "bbox_x1_", "bbox_x2_", "bbox_y1_", "bbox_y2_",
        "irid_mode_radio_", "irid_boost_", "subject_search",
        "illu_", "add_region_", "del_region_",
    )
    for k in list(st.session_state.keys()):
        if any(str(k).startswith(p) for p in prefixes):
            st.session_state.pop(k, None)

    if not keep_login:
        st.session_state.pop(SS_AUTH_USER, None)


def render_action_buttons() -> None:
    """Upload → Convert → Finish 세로 버튼.

    Convert 버튼은 피사체 + 광원 둘 다 선택돼야 활성화된다.
    """
    st.write("")  # 상단 여백

    # ----- Upload -----
    up = st.file_uploader(
        "Upload",
        type=["jpg", "jpeg", "png"],
        key="uploader",
        label_visibility="collapsed",
    )
    if up is not None and st.session_state.get("_last_upload_name") != up.name:
        try:
            handle_image_upload(up)
            st.session_state["_last_upload_name"] = up.name
            st.rerun()
        except NotImplementedError as e:
            st.warning(str(e))

    st.write("")

    # ----- Convert (gated) -----
    regions: list[dict] = st.session_state.get(SS_REGIONS) or []
    illu = st.session_state.get(SS_SELECTED_ILLU)
    has_image = st.session_state.get(SS_ORIGIN_BYTES) is not None
    has_region = len(regions) > 0
    has_illu = bool(illu)

    convert_disabled = not (has_image and has_region and has_illu)
    convert_help = None
    if not has_image:
        convert_help = "먼저 이미지를 업로드하세요."
    elif not has_region:
        convert_help = "피사체 영역을 1개 이상 추가하세요."
    elif not has_illu:
        convert_help = "광원을 선택하세요."

    if st.button(
        "Convert",
        type="primary",
        **_FW,
        disabled=convert_disabled,
        help=convert_help,
    ):
        _run_convert_multi(regions, illu)

    st.write("")

    # ----- 비우기 (이미지/태그/결과 초기화 · 로그인 유지) -----
    if st.button(
        "🗑️ 비우기",
        **_FW,
        help="이미지·영역·광원·결과를 모두 지웁니다. 로그인은 유지됩니다.",
        disabled=not (has_image or has_region or has_illu),
    ):
        _reset_workspace(keep_login=True)
        st.toast("작업 공간을 비웠습니다.")
        st.rerun()

    st.write("")

    # ----- Finish -----
    finish_disabled = not st.session_state.get(SS_BASE_COLORS)
    if st.button(
        "Finish",
        **_FW,
        disabled=finish_disabled,
        help=("Convert 를 먼저 실행하세요." if finish_disabled else None),
    ):
        # 멀티리전: 첫 영역의 피사체를 "대표" 로 (subject_id 단일 컬럼용),
        # 모든 영역 라벨은 subject_labels JSONB 컬럼으로 함께 저장.
        rep_subj = (regions[0].get("subject") if regions else {}) or {}
        labels = [
            (r.get("subject") or {}).get("label")
            for r in regions
            if r.get("subject")
        ]

        draft_project = {
            "id": None,  # 아직 INSERT 전
            "subject_id": rep_subj.get("id"),
            "illu_id": (illu or {}).get("id"),
            "subject_labels": labels,
            "scrap_count": 0,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "is_anonymous": st.session_state.get(SS_IS_ANON, False),
            "_subject_label": rep_subj.get("label"),
            "_illu_name": (illu or {}).get("name"),
            "_base_colors": st.session_state.get(SS_BASE_COLORS) or [],
            "_origin_bytes": st.session_state.get(SS_ORIGIN_BYTES),
            "_result_bytes": st.session_state.get(SS_RESULT_BYTES),
            "_matrix": st.session_state.get(SS_MATRIX),
            "_regions_meta": [
                {
                    "label": (r.get("subject") or {}).get("label"),
                    "matrix": r.get("matrix"),
                    "mask_px": (int(r["mask"].sum()) if r.get("mask") is not None and np is not None else None),
                }
                for r in regions
            ],
        }
        show_project_modal(draft_project, mode="share")


def _run_convert_multi(regions: list[dict], illu: dict) -> None:
    """멀티리전 Convert 핸들러.

    [핵심 규칙]
      * 광원은 1개 공통, 피사체는 영역별 N개.
      * 시작점은 원본 이미지(`out = img.copy()`) → 안 칠한 영역은 자동으로 원본 유지.
      * 마스크 겹침은 **last-write-wins** (루프 후순위가 덮어쓴다).
      * 한 영역의 AI 응답이 스키마를 위반하면 그 영역만 `IDENTITY` 로 fallback,
        나머지 영역은 정상 처리한다 (부분 실패 격리).
      * 마스크는 절대 DB 에 저장하지 않는다 — session_state 안에서만 휘발.
    """
    origin_bytes = st.session_state.get(SS_ORIGIN_BYTES)
    if not origin_bytes:
        st.error("원본 이미지가 없습니다.")
        return
    if np is None:
        st.error("numpy 가 필요합니다 (requirements 설치).")
        return

    img = load_rgb(origin_bytes)               # H×W×3 float32
    out = img.copy()                            # 시작점 = 원본
    matrices_for_session: list[dict] = []
    fallback_count = 0

    # 광원 SPD 는 이미지 전체 1개 공통 (사진 한 장 = 한 조명).
    spd = (illu or {}).get("spd_data")

    # ── 진단: SPD/reflectance 가 비어있으면 변환이 무력화되므로 미리 경고 ──
    spd_probe = spectrum_to_array(spd)
    if float(spd_probe.sum()) == 0.0:
        st.error(
            f"⚠️ 광원 '{(illu or {}).get('name')}' 의 SPD(spd_data) 가 비어있거나 "
            "파싱되지 않았습니다 → 변환이 적용되지 않습니다. "
            "standard_illuminants 테이블의 spd_data 값을 확인하세요."
        )
        return

    with st.spinner(f"UV false-color 변환 × {len(regions)} 영역…"):
        for region in regions:
            subj = region.get("subject") or {}
            reflectance = subj.get("spectral_values")
            is_uv_active = bool(subj.get("is_uv_active"))

            # ── 분광 반사율 × 광원 SPD → false-color 기준색 (GPT 호출 없음) ──
            # 반사율이 없는 영역만 변환 불가로 간주해 원본 유지(부분 실패 격리).
            try:
                target = compute_false_color_target(reflectance, spd, is_uv_active)
            except Exception as e:  # pragma: no cover - defensive
                st.warning(
                    f"변환 실패 ({subj.get('label')}): {e} → 원본 유지."
                )
                fallback_count += 1
                continue

            if not reflectance:
                # 분광 데이터 없는 피사체 — 칠하지 않고 원본 유지
                fallback_count += 1
                continue

            # region 호환 필드: 기존 스키마(_matrix/matrix)와의 연속성을 위해
            # 산출한 target RGB 와 대역 진단을 region["matrix"] 에 dict 로 보관한다.
            region_payload = {
                "target_rgb": target["rgb"],
                "bands": target["bands"],
                "uv_fraction": target["uv_fraction"],
                "illuminant": (illu or {}).get("name"),
            }
            region["matrix"] = region_payload
            matrices_for_session.append(region_payload)

            try:
                transformed = apply_false_color_array(
                    img,
                    target["rgb"],
                    strength=1.0,
                    mode=region.get("irid_mode", "iridescent"),
                    sat_boost=float(region.get("irid_boost", 7.0)),
                )
            except NotImplementedError:
                st.error("PHASE 3 (numpy 변환) 의존성이 미설치 상태입니다.")
                return

            mask = region.get("mask")
            if mask is None:
                # 마스크 없음 = 전체 적용 (last-write-wins 규칙상 이전 영역을 덮음)
                out = transformed
            else:
                # 마스크 영역만 덮어쓰기 — 안 칠한 부분은 그대로 유지
                out[mask] = transformed[mask]

    result_bytes = to_png_bytes(out)
    st.session_state[SS_RESULT_BYTES] = result_bytes
    # 대표 페이로드는 첫 영역 (디버그/표시용) — 합성 결과 재현은 어차피 마스크 없이 불가
    st.session_state[SS_MATRIX] = (
        matrices_for_session[0] if matrices_for_session else {}
    )

    try:
        colors = extract_base_colors(result_bytes, k=5)
    except NotImplementedError:
        colors = []
    st.session_state[SS_BASE_COLORS] = colors
    st.session_state[SS_TILE_GRID] = build_5x10_tile_grid(colors)

    if fallback_count:
        st.warning(
            f"⚠️ {fallback_count}개 영역이 분광 데이터(spectral_values) 부재로 "
            f"변환되지 않고 원본 유지됨."
        )
    st.success(f"✅ 변환 완료 · {len(regions)}개 영역 합성")
    st.rerun()


# ---------- 모달: 설정 (⚙️) — 닉네임 편집 -------------------------------------
@st.dialog("⚙️ 설정", width="small")
def show_settings_modal() -> None:
    """사이드바 ⚙️ 버튼이 여는 설정 모달.

    현재 지원: 닉네임 변경 (`profiles.nickname` UPDATE/UPSERT).
    저장 시 `SS_AUTH_USER.nickname` 도 함께 갱신하여 사이드바 표시가 즉시 반영된다.
    """
    user = ensure_authenticated()
    if not user:
        st.warning("로그인이 필요합니다.")
        return

    st.markdown(f"**계정**: `{user.get('email') or '-'}`")
    st.caption(f"@{(user.get('id') or '')[:8]}")
    st.divider()

    current_nick = user.get("nickname") or ""
    new_nick = st.text_input(
        "닉네임",
        value=current_nick,
        key="settings_nickname_input",
        max_chars=50,
    )

    btn_cols = st.columns([1, 1])
    if btn_cols[0].button("취소", key="settings_cancel", **_FW):
        st.rerun()
    if btn_cols[1].button(
        "저장",
        key="settings_save",
        type="primary",
        **_FW,
        disabled=(new_nick.strip() == current_nick.strip()),
    ):
        msg = update_user_nickname(user["id"], new_nick)
        if msg.startswith("✅"):
            # 사이드바 즉시 반영
            st.session_state[SS_AUTH_USER] = {
                **user,
                "nickname": new_nick.strip(),
            }
            st.success(msg)
            st.rerun()
        else:
            st.error(msg)

    # ── 진단 패널 ─────────────────────────────────────────────────────────
    st.divider()
    with st.expander("🔧 연결 / AI 진단", expanded=False):
        st.markdown("**Supabase**")
        st.text(f"URL: {s_url or '(미설정)'}")
        st.text(f"user_id (full): {user.get('id') or '(없음)'}")
        st.caption(
            "Supabase Dashboard 의 URL 과 비교하세요. "
            "다르면 다른 프로젝트에 가입/저장되고 있는 것입니다."
        )

        st.markdown("**OpenAI API 키**")
        ki = _openai_key_info()
        if not ki["source"]:
            st.error(
                "키가 로드되지 않았습니다.\n"
                "- 환경변수: `OPENAI_API_KEY=sk-...`\n"
                "- 또는 `.streamlit/secrets.toml` 의 `[openai] api_key = \"sk-...\"`"
            )
        else:
            st.text(f"source: {ki['source']}  length: {ki['length']}")
            st.text(f"preview: {ki['preview']}")

        st.markdown("**마지막 GPT-4o-mini 호출**")
        diag = st.session_state.get(SS_OPENAI_LAST_DIAG)
        if not diag:
            st.caption("아직 호출 기록이 없습니다. Convert 를 한 번 실행해 보세요.")
        else:
            kind = "✅" if diag.get("ok") else "❌"
            st.text(f"{kind} stage: {diag.get('stage')}  ({diag.get('ts', '-')})")
            if diag.get("detail"):
                st.text(f"detail: {diag['detail']}")
            # 자주 걸리는 패턴은 한 줄로 강조 (insufficient_quota 등).
            hint = _format_openai_error_hint(diag.get("detail"))
            if hint:
                st.warning(hint)
            if diag.get("raw_preview"):
                st.code(diag["raw_preview"], language="json")


# ---------- 모달 (조회 / 공유 확정 — 같은 컴포넌트 재사용) ----------------------
@st.dialog("프로젝트 미리보기", width="large")
def show_project_modal(project: dict, mode: str = "view") -> None:
    """`mode='view'` : 라인업 카드 클릭 시 조회.
    `mode='share'`: Finish 버튼 클릭 시 저장+공유 확정 흐름.

    같은 모달 컴포넌트를 두 모드로 재사용한다.
    """
    is_share = mode == "share"

    # ---- 상단: 원본 + 5×10 팔레트 미리보기 ----
    left, right = st.columns([1, 1])
    with left:
        st.markdown("**원본 이미지**")
        if is_share and project.get("_origin_bytes"):
            st.image(project["_origin_bytes"], **_FW)
        elif project.get("origin_path"):
            st.image(project["origin_path"], **_FW)
        else:
            st.caption("(원본 이미지 없음)")

    with right:
        st.markdown("**5×10 팔레트**")
        if is_share:
            base_colors = project.get("_base_colors") or []
        else:
            base_colors = fetch_palette_for_project(project["id"]) if project.get("id") else []
        grid = build_5x10_tile_grid(base_colors)
        if grid:
            for row in grid:
                row_html = "".join(
                    f"<div style='flex:1;height:18px;background:{c};'></div>" for c in row
                )
                st.markdown(
                    f"<div style='display:flex;gap:1px;margin-bottom:1px;'>{row_html}</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("(팔레트 없음)")

    st.divider()

    # ---- 메타: 태그 / 스크랩 수 / 공유일 / 제작자 ----
    meta_cols = st.columns(4)

    # 피사체 = N개 라벨 배열 (멀티리전). 단일 라벨로 표시할 수 없으므로 join.
    labels: list[str] = list(project.get("subject_labels") or [])
    if not labels:
        # back-compat: 구 단일 컬럼 fallback
        if is_share and project.get("_subject_label"):
            labels = [project["_subject_label"]]
        elif project.get("subject_id"):
            fallback = _lookup_label("reflectance_library", project["subject_id"])
            labels = [fallback] if fallback else []
    subj_display = ", ".join(labels) if labels else "—"

    if is_share:
        illu_name = project.get("_illu_name")
    else:
        illu_name = _lookup_label("standard_illuminants", project.get("illu_id"), name_col="name")

    meta_cols[0].metric("피사체", subj_display)
    meta_cols[1].metric("광원", illu_name or "—")
    meta_cols[2].metric("스크랩", project.get("scrap_count", 0))
    meta_cols[3].metric(
        "공유일",
        (project.get("created_at") or "")[:10] or "—",
    )

    creator = "익명의 창작자" if project.get("is_anonymous") else (project.get("owner_id") or "—")
    st.caption(f"제작자: {creator}")

    # ---- 모드별 추가 액션 ----
    st.divider()
    if is_share:
        st.markdown("### 공유 옵션")
        anon = st.checkbox(
            "닉네임 비공개 (익명의 창작자로 공유)",
            value=st.session_state.get(SS_IS_ANON, False),
            key="modal_is_anon",
        )
        pal_pub = st.checkbox(
            "팔레트 공개 (다른 사용자가 5×10 을 볼 수 있도록)",
            value=st.session_state.get(SS_IS_PALETTE_PUBLIC, True),
            key="modal_is_palette_public",
        )

        btn_cols = st.columns([1, 1])
        if btn_cols[0].button("취소", **_FW, key="modal_cancel"):
            st.rerun()
        if btn_cols[1].button(
            "✅ 공유 확정",
            type="primary",
            **_FW,
            key="modal_confirm_share",
        ):
            st.session_state[SS_IS_ANON] = anon
            st.session_state[SS_IS_PALETTE_PUBLIC] = pal_pub

            # ── 원본/결과 이미지를 Supabase Storage 에 업로드 ────────────────
            #   /originals/{uuid}/origin.png
            #   /results/{uuid}/result.png
            # 같은 UUID 를 prefix 로 묶어 한 프로젝트의 원본↔결과 매핑을 보존한다.
            doc_uuid = str(uuid.uuid4())
            origin_bytes = project.get("_origin_bytes")
            result_bytes = project.get("_result_bytes")

            origin_url: str | None = None
            result_url: str | None = None
            if origin_bytes:
                try:
                    origin_url = upload_to_storage(
                        BUCKET_ORIGINALS,
                        f"{doc_uuid}/origin.png",
                        origin_bytes,
                    )
                except Exception as e:
                    st.warning(f"원본 Storage 업로드 실패: {e}")
            if result_bytes:
                try:
                    result_url = upload_to_storage(
                        BUCKET_RESULTS,
                        f"{doc_uuid}/result.png",
                        result_bytes,
                    )
                except Exception as e:
                    st.warning(f"결과 Storage 업로드 실패: {e}")

            if not origin_url:
                # origin_path 컬럼은 NOT NULL 이므로 최소한 placeholder 라도 채운다.
                origin_url = f"local://{doc_uuid}/origin.png"

            user = ensure_authenticated() or {}
            msg = save_simulation_result(
                user_id=user.get("id"),
                matrix=project.get("_matrix") or {},
                origin_url=origin_url,
                base_colors=project.get("_base_colors") or [],
                subject_id=project.get("subject_id"),
                subject_labels=project.get("subject_labels") or labels or None,
                illu_id=project.get("illu_id"),
                result_url=result_url,
                is_anonymous=anon,
                is_palette_public=pal_pub,
            )
            st.success(msg)
            st.rerun()
    else:
        # view 모드 — 스크랩 / 삭제 요청
        if project.get("status") == "Pending_Deletion":
            st.warning(
                "⚠️ 이 프로젝트는 **삭제 예정** 상태입니다. "
                "신규 스크랩이 비활성화되며, 잠금 해제 시 자동 삭제됩니다."
            )
            st.button("📌 스크랩하기", disabled=True, key=f"modal_scrap_{project.get('id')}")
        else:
            if st.button("📌 스크랩하기", key=f"modal_scrap_{project.get('id')}", type="primary"):
                msg = scrap_project(project["id"], project.get("scrap_count", 0))
                st.toast(msg)
                st.rerun()


def _lookup_label(table: str, row_id: str | None, name_col: str = "label") -> str | None:
    if not row_id:
        return None
    try:
        resp = (
            conn.table(table).select(f"id, {name_col}").eq("id", row_id).limit(1).execute()
        )
        if resp.data:
            return resp.data[0].get(name_col)
    except Exception:
        pass
    return None


# =============================================================================
# UI ENTRY POINT
# =============================================================================

def main() -> None:
    # NOTE: st.set_page_config() 는 모듈 최상단에서 한 번만 호출 (중복 호출 금지)
    st.title("🎨 UV 모사 팔레트 연구실")

    render_sidebar()

    # 본문 3-컬럼: [중앙 원본+태그] [버튼 세로 스택] [우측 결과+5x10]
    col_center, col_buttons, col_right = st.columns([3, 0.9, 2.5])

    with col_center:
        render_original_image_panel()
        render_tag_panel()

    with col_buttons:
        render_action_buttons()

    with col_right:
        render_converted_image_panel()
        render_5x10_grid()

    # 개발자/회귀 점검용 (Legacy)
    with st.expander("🔧 Legacy Test Panel (개발용 — 원본 app.py 회귀 확인)"):
        _render_legacy_test_panel()


def _render_legacy_test_panel() -> None:
    """기존 app.py 의 동작 확인용 패널 (회귀 방지)."""
    target_label = st.text_input(
        "조회할 피사체 라벨을 입력하세요 (예: Mallard Duck)",
        "Mallard Duck",
        key="legacy_label",
    )
    if st.button("DB에서 물리 데이터 가져오기", key="legacy_fetch"):
        data = get_reflectance_data(target_label)
        if data:
            st.success(f"'{target_label}' 데이터를 찾았습니다!")
            st.json(data)
        else:
            st.warning("데이터가 없습니다. SQL Editor에서 데이터를 먼저 넣어주세요!")

    if st.button("샘플 프로젝트 저장 시뮬레이션", key="legacy_save"):
        test_result = save_simulation_result(
            user_id=None,
            matrix={"r": [1, 0, 0, 0], "g": [0, 1, 0, 0], "b": [0, 0, 1, 0]},
            origin_url="https://example.com/test.jpg",
            base_colors=["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#00FFFF"],
        )
        st.info(test_result)


if __name__ == "__main__":
    main()
