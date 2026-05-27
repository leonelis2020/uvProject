"""
UV 모사 팔레트 연구실 (UV Simulation Palette Lab)
==================================================
Fat-Client (Streamlit) + Thin-DB (Supabase) 스카폴딩.

설계 원칙
---------
* 5x10 명채도 매트릭스 타일과 같은 휘발성/연산 집약적 데이터는 DB에 저장하지 않고,
  프론트엔드(Streamlit)에서 실시간 연산한다. DB는 `matrix_json`(3x4) + 대표 색상
  `base_colors`(5 HEX)만 보관한다.
* RAG (Retrieval Augmented Generation) 컨텍스트는 `standard_illuminants`와
  `reflectance_library` 두 마스터 테이블에서 조회하여 GPT-4o 프롬프트에 결합한다.
* 모든 사용자 산출물(`projects`)은 덮어쓰지 않고 `parent_id` 셀프 참조 스냅샷을 통해
  버전 히스토리를 유지한다. 스크랩(reuse) 역시 `parent_id`로 부모 프로젝트를 가리킨다.

이 파일은 "큰 틀(Scaffolding)"이며, 각 단계(PHASE 1~5)는 향후 세부 알고리즘
(OpenCV 변환, drawable-canvas 마스킹, GPT-4o 멀티모달 호출 등)이 살을 붙일 수
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

# =============================================================================
# PHASE 0: CONNECTION & MASTER LAYER
#   - 시스템 환경 변수 → secrets.toml 백업 순으로 자격 증명 로드 (기존 로직 보호)
#   - `standard_illuminants`, `reflectance_library` 마스터 데이터 접근 헬퍼
# =============================================================================

# 1. Supabase 연결 설정 (secrets.toml 필요)
# 파일이 아니라 시스템(코드스페이스) 환경 변수에서 우선 가져옴
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
    """RAG용 반사율 라이브러리 조회.

    기존 시그니처를 그대로 유지한다. 향후 `category` 필터, `is_uv_active` 토글,
    파장 범위 슬라이싱 등을 인자로 받아 확장될 수 있다.
    """
    query = (
        conn.table("reflectance_library")
        .select("*")
        .eq("label", label)
        .execute()
    )
    return query.data if query.data else None


def get_illuminant_spd(name: str) -> dict | None:
    """광원(`standard_illuminants`) SPD(분광 전력 분포) 조회.

    GPT-4o 프롬프트 합성 시 `reflectance_library.spectral_values`와 곱해
    조명 가중 색상을 추정하는 데 사용된다.
    """
    query = (
        conn.table("standard_illuminants")
        .select("*")
        .eq("name", name)
        .single()
        .execute()
    )
    return query.data if getattr(query, "data", None) else None


# =============================================================================
# PHASE 1: INPUT  —  이미지 업로드 / 리사이즈 / 마스킹
#   - PIL.Image.thumbnail((1024, 1024)) 1024px 리사이즈
#   - streamlit-drawable-canvas 마스크 UI 슬롯
#   - 결과는 st.session_state 에 저장하여 PHASE 2 가 소비한다.
# =============================================================================

# session_state 키 표준 (모듈 간 계약)
SS_ORIGIN_IMAGE = "origin_image_pil"     # PIL.Image — 1024px 리사이즈 본
SS_ORIGIN_BYTES = "origin_image_bytes"   # bytes — Storage 업로드용 원본
SS_MASK_ARRAY = "mask_array"             # np.ndarray — drawable-canvas 결과
SS_MATRIX = "matrix_json"                # dict — GPT-4o 가 반환한 3x4 매트릭스
SS_BASE_COLORS = "base_colors"           # list[str] — HEX 5개
SS_TILE_GRID = "tile_grid_5x10"          # list[list[str]] — 휘발성, DB 미저장
SS_IS_ANON = "is_anon"                   # bool — 익명 공유 토글
SS_AUTH_USER = "auth_user"               # dict | None — Supabase Auth user


def handle_image_upload(uploaded_file) -> None:
    """업로드 파일을 1024px 썸네일로 리사이즈하고 session_state 에 저장.

    NOTE (PHASE 1 확장 지점):
      * 향후 EXIF 회전 보정, sRGB 프로파일 강제 변환, HEIC 디코딩 등은
        본 함수 내부에서 처리하라.
      * 결과물의 메타데이터(width, height, channels)도 session_state 에
        함께 저장하여 PHASE 2 의 프롬프트 토큰 절약 로직에 활용 가능.
    """
    # TODO[PHASE-1]: from PIL import Image; img = Image.open(uploaded_file)
    # TODO[PHASE-1]: img.thumbnail((1024, 1024)); buf = io.BytesIO(); img.save(buf, "PNG")
    # TODO[PHASE-1]: st.session_state[SS_ORIGIN_IMAGE] = img
    # TODO[PHASE-1]: st.session_state[SS_ORIGIN_BYTES] = buf.getvalue()
    raise NotImplementedError("PHASE 1: PIL 리사이즈 로직 미구현 (스카폴딩 슬롯)")


def render_mask_canvas() -> None:
    """streamlit-drawable-canvas 위젯을 렌더링하고 마스크를 session_state 에 저장.

    NOTE (PHASE 1 확장 지점):
      * `from streamlit_drawable_canvas import st_canvas` 후
        `background_image = st.session_state[SS_ORIGIN_IMAGE]`,
        `drawing_mode="freedraw"`, `stroke_width=20`, `update_streamlit=True`
        설정으로 호출하라.
      * `st_canvas` 의 `image_data[:,:,3]` 알파 채널을 0/255 이진화하여
        `SS_MASK_ARRAY` 로 저장하면 PHASE 3 의 cv2.transform 마스킹과 호환된다.
    """
    st.info("🖌️ (PHASE 1) drawable-canvas 마스크 UI 가 여기에 렌더링될 예정입니다.")
    # TODO[PHASE-1]: canvas_result = st_canvas(...)
    # TODO[PHASE-1]: st.session_state[SS_MASK_ARRAY] = binarize(canvas_result.image_data)


def render_phase1_input_panel() -> None:
    """PHASE 1 의 통합 UI 패널 (업로드 + 캔버스)."""
    st.subheader("1️⃣ 이미지 입력 & 마스킹")
    up = st.file_uploader("원본 이미지 업로드 (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if up is not None:
        try:
            handle_image_upload(up)
        except NotImplementedError as e:
            st.warning(str(e))
    render_mask_canvas()


# =============================================================================
# PHASE 2: AI SIMULATION  —  RAG + GPT-4o 멀티모달 호출
#   - reflectance + illuminant 컨텍스트 조립 → 프롬프트 합성
#   - GPT-4o 호출 → 3x4 변환 행렬(JSON) 수신 및 스키마 검증
# =============================================================================

# 3x4 매트릭스 JSON 의 기대 스키마 (수신 검증용)
EXPECTED_MATRIX_KEYS = {"r", "g", "b"}
EXPECTED_MATRIX_ROW_LEN = 4  # [r0, r1, r2, bias]


def build_rag_context(target_label: str, illuminant_name: str) -> dict:
    """GPT-4o 프롬프트에 결합할 RAG 컨텍스트 패키지."""
    reflectance = get_reflectance_data(target_label) or []
    illuminant = get_illuminant_spd(illuminant_name) or {}
    return {
        "target_label": target_label,
        "illuminant": illuminant,
        "reflectance_samples": reflectance,
    }


def validate_matrix_payload(payload: Any) -> bool:
    """GPT-4o 가 반환한 JSON 이 3x4 매트릭스 규약을 따르는지 검증."""
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


def call_gpt4o_for_matrix(rag_context: dict, image_bytes: bytes) -> dict:
    """GPT-4o 멀티모달 호출 → 3x4 변환 행렬 JSON.

    NOTE (PHASE 2 확장 지점):
      * `from openai import OpenAI; client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])`
      * `client.chat.completions.create(model="gpt-4o", response_format={"type":"json_object"}, ...)`
      * 시스템 프롬프트에 `rag_context` 를 JSON 직렬화하여 주입하라.
      * 반환 직후 반드시 `validate_matrix_payload()` 로 검증하고,
        실패 시 1회 retry → 그래도 실패하면 항등행렬을 fallback 으로 반환.
    """
    # TODO[PHASE-2]: 실제 OpenAI 호출 구현
    raise NotImplementedError("PHASE 2: GPT-4o 호출 로직 미구현 (스카폴딩 슬롯)")


def render_phase2_simulation_panel(illuminant_name: str) -> None:
    """PHASE 2 의 UI 패널 (RAG 미리보기 + 시뮬레이션 트리거)."""
    st.subheader("2️⃣ RAG 컨텍스트 & AI 시뮬레이션")
    target_label = st.text_input(
        "조회할 피사체 라벨 (예: Mallard Duck)",
        "Mallard Duck",
        key="phase2_label",
    )

    if st.button("🔍 RAG 컨텍스트 미리보기"):
        ctx = build_rag_context(target_label, illuminant_name)
        st.json(ctx)

    if st.button("🤖 GPT-4o 시뮬레이션 실행"):
        ctx = build_rag_context(target_label, illuminant_name)
        try:
            matrix = call_gpt4o_for_matrix(ctx, st.session_state.get(SS_ORIGIN_BYTES, b""))
            if not validate_matrix_payload(matrix):
                st.error("⚠️ GPT-4o 응답이 3x4 매트릭스 스키마를 위반했습니다.")
                return
            st.session_state[SS_MATRIX] = matrix
            st.success("✅ 3x4 매트릭스 수신 완료")
            st.json(matrix)
        except NotImplementedError as e:
            st.warning(str(e))


# =============================================================================
# PHASE 3: LOCAL PIXEL TRANSFORM  —  OpenCV + 5색 추출 + 5x10 명채도 타일
#   - cv2.transform() 으로 마스크 영역에만 3x4 행렬 적용
#   - K-Means 등으로 대표 5색 추출
#   - HSL 변환 후 10% 단위 명도 가공 루프 → 5x10 = 50 타일 (휘발성, DB 미저장)
# =============================================================================


def apply_matrix_to_masked_region(
    image_bytes: bytes,
    mask_array,  # np.ndarray | None
    matrix: dict,
) -> bytes:
    """cv2.transform 으로 마스크 영역에만 3x4 변환을 적용.

    NOTE (PHASE 3 확장 지점):
      * `import cv2, numpy as np`
      * `img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)`
      * `M = np.array([matrix["r"], matrix["g"], matrix["b"]], dtype=np.float32)`  # (3,4)
      * `transformed = cv2.transform(img, M)`
      * `out = np.where(mask_array[..., None] > 0, transformed, img)`
      * return cv2.imencode(".png", out)[1].tobytes()
    """
    # TODO[PHASE-3]: 실제 OpenCV 연산 구현
    raise NotImplementedError("PHASE 3: cv2.transform 미구현 (스카폴딩 슬롯)")


def extract_base_colors(image_bytes: bytes, k: int = 5) -> list[str]:
    """결과 이미지에서 K-Means 로 대표 K개 HEX 색상 추출."""
    # TODO[PHASE-3]: from sklearn.cluster import KMeans 또는 cv2.kmeans 사용
    raise NotImplementedError("PHASE 3: 대표 색상 추출 미구현 (스카폴딩 슬롯)")


def build_5x10_tile_grid(base_colors: list[str]) -> list[list[str]]:
    """5색 × 10단계 명도 매트릭스(5x10)를 HSL 변환 기반으로 생성.

    이 결과는 **DB 에 저장하지 않는** 휘발성 데이터다 (설계 원칙).
    매번 `base_colors` 로부터 재계산되며 session_state[SS_TILE_GRID] 에만 캐싱된다.

    NOTE (PHASE 3 확장 지점):
      * `import colorsys` 로 HEX→RGB→HLS 변환 후 L 채널을 0.05~0.95 범위에서
        10단계로 균등 샘플링하여 HEX 로 재인코딩하라.
    """
    # TODO[PHASE-3]: HSL 명도 루프 구현
    raise NotImplementedError("PHASE 3: 5x10 타일 생성 미구현 (스카폴딩 슬롯)")


def render_phase3_transform_panel() -> None:
    """PHASE 3 의 UI 패널 (변환 결과 + 5x10 타일 미리보기)."""
    st.subheader("3️⃣ 로컬 픽셀 변환 & 5×10 명채도 타일")
    if st.session_state.get(SS_MATRIX) is None:
        st.info("PHASE 2 에서 매트릭스를 먼저 수신해야 합니다.")
        return

    if st.button("🎨 변환 실행"):
        try:
            result_bytes = apply_matrix_to_masked_region(
                st.session_state.get(SS_ORIGIN_BYTES, b""),
                st.session_state.get(SS_MASK_ARRAY),
                st.session_state[SS_MATRIX],
            )
            colors = extract_base_colors(result_bytes, k=5)
            st.session_state[SS_BASE_COLORS] = colors
            st.session_state[SS_TILE_GRID] = build_5x10_tile_grid(colors)
            st.success("✅ 변환 및 타일 생성 완료")
        except NotImplementedError as e:
            st.warning(str(e))

    # 5x10 그리드 렌더링 (휘발성, DB 미저장)
    grid = st.session_state.get(SS_TILE_GRID)
    if grid:
        st.caption("5×10 명채도 매트릭스 (휘발성 — DB 미저장)")
        for row in grid:
            cols = st.columns(len(row))
            for c, hex_code in zip(cols, row):
                c.markdown(
                    f"<div style='background:{hex_code};height:32px;border-radius:4px;'></div>",
                    unsafe_allow_html=True,
                )


# =============================================================================
# PHASE 4: PERSISTENCE & STORAGE
#   - Supabase Auth 세션 체크
#   - Storage 버킷: /originals/ (원본) , /results/ (변환 결과) 분리
#   - DB INSERT 트랜잭션: projects → palettes (FK: palettes.project_id)
#     * 비즈니스 규칙: palettes 는 projects 보다 먼저 만들 수 없다.
#       projects ON DELETE CASCADE 로 palettes 가 자동 삭제됨.
#     * parent_id (projects 셀프 FK) 는 수정 스냅샷 / 스크랩 복제 시 채워짐.
# =============================================================================

BUCKET_ORIGINALS = "originals"
BUCKET_RESULTS = "results"


def ensure_authenticated() -> dict | None:
    """Supabase Auth 세션을 확인하고 user 객체를 반환.

    NOTE (PHASE 4 확장 지점):
      * `conn.auth.get_user()` 또는 `st_supabase_connection` 의 OAuth 위젯으로 대체.
      * 비로그인 시 None 반환 + UI 단에서 로그인 버튼 노출.
    """
    return st.session_state.get(SS_AUTH_USER)


def upload_to_storage(bucket: str, path: str, data: bytes) -> str:
    """Supabase Storage 업로드 후 public URL (또는 signed URL) 반환."""
    # TODO[PHASE-4]: conn.client.storage.from_(bucket).upload(path, data)
    # TODO[PHASE-4]: return conn.client.storage.from_(bucket).get_public_url(path)
    raise NotImplementedError("PHASE 4: Storage 업로드 미구현 (스카폴딩 슬롯)")


def save_simulation_result(user_id, matrix, origin_url, base_colors):
    """프로젝트 + 팔레트 동시 저장 (기존 시그니처 보호).

    [데이터 무결성 규칙]
      * projects 가 먼저 INSERT 되어 `id` 가 확정되어야 palettes.project_id 가
        충족된다. 두 호출 사이에서 예외 발생 시 projects 만 남는 고아 레코드가
        생길 수 있으므로 향후 RPC(트랜잭션 함수) 로 승격할 것.
      * `parent_id` 셀프 FK 는 본 함수에서는 NULL 이며, 수정/스크랩 흐름에서
        `save_snapshot()` / `scrap_project()` 가 채워준다.
      * `palettes` 는 projects 에 대해 ON DELETE CASCADE 이므로 별도 정리 불필요.
    """
    # (1) 데이터 준비 및 실행
    data_to_insert = {
        "owner_id": user_id,
        "matrix_json": matrix,
        "origin_path": origin_url,
        "is_anonymous": st.session_state.get(SS_IS_ANON, False),
        "status": "Active",
        "scrap_count": 0,
    }

    project_resp = conn.table("projects").insert(data_to_insert).execute()

    # (2) 결과가 있으면 ID 추출 → palettes 연쇄 INSERT
    if project_resp.data:
        try:
            row = project_resp.data[0] if isinstance(project_resp.data, list) else project_resp.data
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
    # TODO[PHASE-4]: 위 save_simulation_result 와 유사하나 parent_id 필드 추가
    raise NotImplementedError("PHASE 4: 스냅샷 INSERT 미구현 (스카폴딩 슬롯)")


def render_phase4_save_panel() -> None:
    """PHASE 4 의 UI 패널 (로그인 체크 + 저장)."""
    st.subheader("4️⃣ 영속성 — Storage 업로드 & DB 저장")
    user = ensure_authenticated()
    st.session_state[SS_IS_ANON] = st.checkbox(
        "익명의 창작자로 공유", value=st.session_state.get(SS_IS_ANON, False)
    )

    if st.button("💾 결과 저장"):
        if user is None and not st.session_state.get(SS_IS_ANON):
            st.error("로그인이 필요합니다. (또는 익명 공유를 선택하세요)")
            return
        try:
            # TODO[PHASE-4]: 실제 Storage 업로드로 교체
            origin_url = "https://example.com/origin.png"
            msg = save_simulation_result(
                user_id=(user or {}).get("id"),
                matrix=st.session_state.get(SS_MATRIX, {"r": [1, 0, 0, 0], "g": [0, 1, 0, 0], "b": [0, 0, 1, 0]}),
                origin_url=origin_url,
                base_colors=st.session_state.get(SS_BASE_COLORS, []),
            )
            st.info(msg)
        except Exception as e:
            st.error(f"저장 실패: {e}")


# =============================================================================
# PHASE 5: BUSINESS LOGIC  —  갤러리 / 스크랩 / 조건부 삭제
#   - '익명의 창작자' DB View 호출 (예: anon_gallery_view)
#   - 스크랩 시 복제본 INSERT (parent_id) + 원본 scrap_count += 1
#   - 조건부 삭제: scrap_count >= 1 OR (now - created_at) < 30일  →
#       즉시 삭제 불가, status = 'Pending_Deletion' 로 전이 + 배너 노출
# =============================================================================

SHARE_LOCK_DAYS = 30  # 공유 후 즉시 삭제 잠금 기간


def fetch_anon_gallery(limit: int = 20) -> list[dict]:
    """'익명의 창작자' 갤러리 피드 (DB View 호출)."""
    # TODO[PHASE-5]: conn.table("anon_gallery_view").select("*").limit(limit).execute()
    try:
        resp = (
            conn.table("anon_gallery_view")
            .select("*")
            .limit(limit)
            .execute()
        )
        return resp.data or []
    except Exception:
        return []


def scrap_project(source_project_id: str, new_owner_id: str) -> str | None:
    """타인의 프로젝트를 스크랩(복제) — parent_id 로 부모를 가리키는 새 row 생성.

    [비즈니스 규칙]
      * 원본 projects.scrap_count += 1 (RPC 또는 SELECT-then-UPDATE).
      * 새 row 는 owner_id = 스크랩한 사용자, parent_id = source_project_id.
      * 원본 status 가 'Pending_Deletion' 이면 스크랩 불가 (UI 단에서 버튼 비활성).
    """
    # TODO[PHASE-5]: SELECT source → INSERT clone → UPDATE source.scrap_count
    raise NotImplementedError("PHASE 5: 스크랩 로직 미구현 (스카폴딩 슬롯)")


def request_project_deletion(project_id: str, created_at_iso: str, scrap_count: int) -> str:
    """삭제 요청 처리.

    [조건부 분기]
      * scrap_count >= 1  →  즉시 삭제 불가 (다른 사용자가 스크랩 사용 중)
      * 공유 후 30일 미만  →  즉시 삭제 불가 (공유 잠금 기간)
      * 둘 다 해당 없음   →  실제 DELETE (CASCADE 로 palettes 동반 삭제)
      * 위 두 조건 중 하나라도 걸리면 status='Pending_Deletion' 로 전이하고
        UI 에 '삭제 예정' 배너 노출.
    """
    try:
        created_at = datetime.fromisoformat(created_at_iso.replace("Z", "+00:00"))
    except Exception:
        created_at = datetime.utcnow()

    locked_by_scrap = scrap_count >= 1
    locked_by_period = (datetime.utcnow() - created_at.replace(tzinfo=None)) < timedelta(days=SHARE_LOCK_DAYS)

    if locked_by_scrap or locked_by_period:
        # 조건부 삭제 대기 상태로 전이
        try:
            conn.table("projects").update({"status": "Pending_Deletion"}).eq("id", project_id).execute()
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


def render_pending_deletion_banner(project: dict) -> None:
    """status == 'Pending_Deletion' 인 프로젝트에 대해 배너 + 스크랩 버튼 비활성."""
    if project.get("status") == "Pending_Deletion":
        st.warning(
            "⚠️ 이 프로젝트는 **삭제 예정** 상태입니다. "
            "신규 스크랩이 비활성화되며, 잠금 해제 시 자동 삭제됩니다."
        )
        st.button(
            "📌 스크랩하기",
            key=f"scrap_{project.get('id', uuid.uuid4())}",
            disabled=True,
            help="삭제 예정 프로젝트는 스크랩할 수 없습니다.",
        )
    else:
        if st.button("📌 스크랩하기", key=f"scrap_{project.get('id', uuid.uuid4())}"):
            try:
                scrap_project(project["id"], (ensure_authenticated() or {}).get("id"))
                st.success("스크랩 완료")
            except NotImplementedError as e:
                st.warning(str(e))


def render_phase5_gallery_panel() -> None:
    """PHASE 5 의 UI 패널 (갤러리 + 스크랩 + 삭제 요청)."""
    st.subheader("5️⃣ 익명의 창작자 갤러리")
    items = fetch_anon_gallery(limit=12)
    if not items:
        st.caption("아직 공유된 프로젝트가 없거나 anon_gallery_view 가 비어 있습니다.")
        return
    for proj in items:
        with st.container(border=True):
            st.markdown(f"**프로젝트** `{proj.get('id', '')[:8]}…`")
            cols = st.columns([3, 1])
            with cols[0]:
                st.json({k: proj.get(k) for k in ("status", "scrap_count", "is_anonymous")})
            with cols[1]:
                render_pending_deletion_banner(proj)


# =============================================================================
# UI ENTRY POINT
# =============================================================================

def main() -> None:
    st.set_page_config(page_title="UV 모사 팔레트 연구실", page_icon="🎨", layout="wide")
    st.title("🎨 UV 모사 팔레트 연구실")
    st.write("DB 설계자와 함께 만든 시스템이 작동하는지 확인해봅시다.")

    # 사이드바: 광원 선택 + 인증 placeholder
    st.sidebar.header("⚙️ 설정")
    illuminant = st.sidebar.selectbox(
        "광원을 선택하세요",
        ["D65", "Sunrise", "UV-A"],
        index=0,
    )
    st.sidebar.caption(f"선택된 광원: `{illuminant}`")
    # TODO[PHASE-4]: 여기서 Supabase Auth 로그인 위젯 렌더링

    tabs = st.tabs([
        "1️⃣ Input",
        "2️⃣ AI Simulation",
        "3️⃣ Local Transform",
        "4️⃣ Persist",
        "5️⃣ Gallery",
        "🔧 Legacy Test",
    ])
    with tabs[0]:
        render_phase1_input_panel()
    with tabs[1]:
        render_phase2_simulation_panel(illuminant)
    with tabs[2]:
        render_phase3_transform_panel()
    with tabs[3]:
        render_phase4_save_panel()
    with tabs[4]:
        render_phase5_gallery_panel()
    with tabs[5]:
        _render_legacy_test_panel()


def _render_legacy_test_panel() -> None:
    """기존 app.py 의 동작 확인용 패널 (회귀 방지)."""
    st.subheader("🔧 기존 회귀 테스트 패널")
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
