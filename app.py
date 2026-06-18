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
    """피사체 50종 전체 목록 (중앙 태그 탭에서 사용)."""
    try:
        resp = (
            conn.table("reflectance_library")
            .select("id, category, label, is_uv_active")
            .order("category")
            .order("label")
            .execute()
        )
        return resp.data or []
    except Exception:
        return []


@st.cache_data(ttl=300, show_spinner=False)
def list_illuminants() -> list[dict]:
    """광원 5종 전체 목록."""
    try:
        resp = (
            conn.table("standard_illuminants")
            .select("id, name, description")
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
# PHASE 2: AI SIMULATION  —  RAG + GPT-4o-mini (텍스트 JSON 전용)
#   - reflectance.spectral_values + illuminant.spd_data 만 프롬프트에 결합
#   - 이미지는 보내지 않음 (저비용 + 환각 방지)
#   - GPT-4o-mini 호출 → 3x4 변환 행렬(JSON) 수신 및 스키마 검증
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


def call_gpt4o_mini_for_matrix(rag_context: dict) -> dict:
    """GPT-4o-mini 호출 → 3x4 변환 행렬 JSON. **이미지는 전달하지 않는다.**

    NOTE (PHASE 2 확장 지점):
      * `from openai import OpenAI`
      * `client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY") or st.secrets["openai"]["api_key"])`
      * `client.chat.completions.create(`
            `model="gpt-4o-mini",`
            `response_format={"type":"json_object"},`
            `messages=[{"role":"system","content": SYSTEM_PROMPT},`
            `          {"role":"user","content": json.dumps(rag_context)}])`
      * 반환 직후 반드시 `validate_matrix_payload()` 로 검증하고,
        실패 시 1회 retry → 그래도 실패하면 항등행렬을 fallback 으로 반환.
    """
    # TODO[PHASE-2]: 실제 OpenAI 호출 구현 (gpt-4o-mini, JSON-only, 텍스트 입력)
    raise NotImplementedError("PHASE 2: GPT-4o-mini 호출 로직 미구현 (스카폴딩 슬롯)")


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
    """Supabase Auth 세션 체크."""
    return st.session_state.get(SS_AUTH_USER)


def upload_to_storage(bucket: str, path: str, data: bytes) -> str:
    """Supabase Storage 업로드 후 public URL (또는 signed URL) 반환."""
    # TODO[PHASE-4]: conn.client.storage.from_(bucket).upload(path, data)
    # TODO[PHASE-4]: return conn.client.storage.from_(bucket).get_public_url(path)
    raise NotImplementedError("PHASE 4: Storage 업로드 미구현 (스카폴딩 슬롯)")


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

    [참조 무결성]
      * 기존 row 를 UPDATE 하지 않고 새 row 를 INSERT 한 뒤 `parent_id` 로 부모를
        가리킨다. 갤러리/히스토리는 parent_id 그래프를 따라간다.
    """
    # TODO[PHASE-4]: save_simulation_result 와 유사하나 parent_id 필드 추가
    raise NotImplementedError("PHASE 4: 스냅샷 INSERT 미구현 (스카폴딩 슬롯)")


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

# ---------- 좌측 사이드바: 프로필 + 라인업 ---------------------------------------
def render_sidebar() -> None:
    """좌측: 프로필 + '내가 공유한' / '남이 공유한' 미리보기 라인업.

    프로필 이미지는 현재 보류(요구사항). 닉네임 + UID 만 노출.
    """
    user = ensure_authenticated() or {}
    with st.sidebar:
        col_a, col_b = st.columns([3, 1])
        with col_a:
            st.markdown(f"**{user.get('nickname') or 'user_nickname'}**")
            st.caption(f"@{(user.get('id') or 'user_UID')[:8]}")
        with col_b:
            st.button("⚙️", key="open_settings", help="설정")

        st.divider()

        st.caption("내가 공유한 미리보기")
        _render_lineup(
            fetch_my_shared_projects(user.get("id"), limit=10),
            key_prefix="mine",
        )

        st.divider()
        st.caption("내가 공유받은(남이 공유한) 미리보기")
        _render_lineup(
            fetch_others_shared_projects(user.get("id"), limit=10),
            key_prefix="others",
        )


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
                    use_container_width=True,
                ):
                    show_project_modal(proj, mode="view")


# ---------- 중앙: 원본 이미지 + 태그 패널 ---------------------------------------
def render_original_image_panel() -> None:
    """중앙 상단: 원본 이미지 프리뷰."""
    with st.container(border=True):
        img = st.session_state.get(SS_ORIGIN_IMAGE)
        if img is not None:
            st.image(img, use_column_width=True)
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
        _render_region_panel()
    with tabs[1]:
        _render_illuminant_tag_grid()
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
                use_container_width=True,
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
    if cv2 is None or np is None:
        st.warning("opencv-python-headless / numpy 가 필요합니다 (requirements 설치).")
        return

    try:
        from streamlit_drawable_canvas import st_canvas
    except Exception:
        st.warning("`streamlit-drawable-canvas` 가 설치되지 않았습니다. (requirements 설치 필요)")
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
                use_container_width=True,
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
            st.image(result, use_column_width=True)
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
        use_container_width=True,
        disabled=convert_disabled,
        help=convert_help,
    ):
        _run_convert_multi(regions, illu)

    st.write("")

    # ----- Finish -----
    finish_disabled = not st.session_state.get(SS_BASE_COLORS)
    if st.button(
        "Finish",
        use_container_width=True,
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

    with st.spinner(f"GPT-4o-mini 호출 × {len(regions)} 영역…"):
        for region in regions:
            subj = region.get("subject") or {}
            ctx = build_rag_context(subj, illu)
            try:
                matrix = call_gpt4o_mini_for_matrix(ctx)
            except NotImplementedError:
                st.info("PHASE 2: GPT-4o-mini 호출이 미구현 상태 → IDENTITY 로 진행합니다.")
                matrix = IDENTITY
            except Exception as e:
                st.warning(f"AI 호출 실패 ({subj.get('label')}): {e} → IDENTITY 로 진행.")
                matrix = IDENTITY

            if not validate_matrix_payload(matrix):
                matrix = IDENTITY
                fallback_count += 1

            region["matrix"] = matrix
            matrices_for_session.append(matrix)

            try:
                transformed = apply_matrix_array(img, matrix)
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
    # 대표 행렬은 첫 영역 (디버그/표시용) — 합성 결과 재현은 어차피 마스크 없이 불가
    st.session_state[SS_MATRIX] = (
        matrices_for_session[0] if matrices_for_session else IDENTITY
    )

    try:
        colors = extract_base_colors(result_bytes, k=5)
    except NotImplementedError:
        colors = []
    st.session_state[SS_BASE_COLORS] = colors
    st.session_state[SS_TILE_GRID] = build_5x10_tile_grid(colors)

    if fallback_count:
        st.warning(
            f"⚠️ {fallback_count}개 영역이 스키마 위반으로 IDENTITY fallback 처리됨."
        )
    st.success(f"✅ 변환 완료 · {len(regions)}개 영역 합성")
    st.rerun()


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
            st.image(project["_origin_bytes"], use_column_width=True)
        elif project.get("origin_path"):
            st.image(project["origin_path"], use_column_width=True)
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
        if btn_cols[0].button("취소", use_container_width=True, key="modal_cancel"):
            st.rerun()
        if btn_cols[1].button(
            "✅ 공유 확정",
            type="primary",
            use_container_width=True,
            key="modal_confirm_share",
        ):
            st.session_state[SS_IS_ANON] = anon
            st.session_state[SS_IS_PALETTE_PUBLIC] = pal_pub
            # TODO[PHASE-4]: upload_to_storage 로 원본/결과 업로드 → URL 획득
            origin_url = "https://example.com/origin.png"
            result_url = "https://example.com/result.png"
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
    st.set_page_config(page_title="UV 모사 팔레트 연구실", page_icon="🎨", layout="wide")
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
