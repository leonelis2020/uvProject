# 🎨 UV 모사 팔레트 연구실 (UV Simulation Palette Lab)

Streamlit(Fat-Client) + Supabase(Thin-DB) 기반의 자외선(UV) 모사 팔레트 스튜디오.
본 저장소는 **확장 가능한 뼈대(Scaffolding)** 단계이며, 향후 OpenCV 변환,
`streamlit-drawable-canvas` 마스킹, OpenAI GPT-4o 멀티모달 호출이 살을 붙일 수 있도록
5개 PHASE 로 모듈화된 구조를 제공한다.

---

## 1. 빠른 실행

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 환경변수 또는 .streamlit/secrets.toml 둘 중 하나만 설정하면 됨
export SUPABASE_URL="https://YOUR-PROJECT.supabase.co"
export SUPABASE_KEY="YOUR-ANON-OR-SERVICE-ROLE-KEY"

streamlit run app.py
```

`secrets.toml` 을 쓰려면 `.streamlit/secrets.toml.example` 을
`.streamlit/secrets.toml` 로 복사한 뒤 값을 채워 넣는다.

---

## 2. PHASE 별 코드 구조

| PHASE | 책임                                       | 핵심 함수 (위치)                                                                                            | 외부 의존성                  |
| ----- | ------------------------------------------ | ----------------------------------------------------------------------------------------------------------- | ---------------------------- |
| 0     | Supabase 연결 + 마스터 데이터              | `get_reflectance_data`, `get_illuminant_spd`, `list_subjects`, `list_illuminants`                           | `st_supabase_connection`     |
| 1     | 이미지 업로드 (1024px) + **영역별 마스킹** | `handle_image_upload`, `_render_mask_canvas_for_region`                                                     | `Pillow`, `opencv-python-headless`, `streamlit-drawable-canvas` |
| 2     | RAG + **GPT-4o-mini (텍스트 JSON 전용) × N영역** | `build_rag_context`, `call_gpt4o_mini_for_matrix`, `validate_matrix_payload`, `_run_convert_multi`     | `openai`                     |
| 3     | numpy 색변환 (배열 in/out) + **OKLCH 5×10 타일** | `load_rgb`, `to_png_bytes`, `apply_matrix_array`, `apply_matrix`, `extract_base_colors`, `make_shades`, `build_5x10_tile_grid` | `numpy`, `scikit-learn`, `coloraide` |
| 4     | Storage + DB INSERT (트랜잭션)             | `upload_to_storage`, `save_simulation_result`, `save_snapshot`                                              | `supabase`                   |
| 5     | 갤러리 라인업 / 스크랩(익명 카운트) / 삭제 | `fetch_my_shared_projects`, `fetch_others_shared_projects`, `scrap_project`, `request_project_deletion`     | `supabase`                   |

각 PHASE 의 미구현 함수는 `NotImplementedError` 또는 `TODO[PHASE-N]` 주석으로
정확한 삽입 지점을 표시해 두었다.

### UI 레이아웃 (Streamlit)

```
┌─ st.sidebar ──┐  ┌─ col_center [3] ──────┐  ┌─ col_buttons [.9] ┐  ┌─ col_right [2.5] ─┐
│ 프로필 + ⚙️    │  │ Original Image Preview │  │ Upload            │  │ Converted Preview │
│ 내가 공유한    │  │ ─────────────────────  │  │ Convert (gated)   │  │ ───────────────── │
│ 라인업 (2col)  │  │ Tabs: 피사체 / 광원   │  │ Finish (modal)    │  │ 5 × 10 그리드     │
│ 남이 공유한    │  │  - 검색 + 단일선택     │  │                   │  │  (OKLCH, 휘발성)  │
│ 라인업 (2col)  │  │                       │  │                   │  │                   │
└────────────────┘  └────────────────────────┘  └───────────────────┘  └───────────────────┘
```

* **Convert 버튼**은 `이미지 업로드 + 피사체 영역 ≥ 1 + 광원 선택` 셋이 모두 충족되어야 활성화된다 (`render_action_buttons` 참조).
* **모달**은 단일 컴포넌트 `show_project_modal(project, mode)` 를 `view` / `share` 두 모드로 재사용한다 (`@st.dialog`).

### 멀티리전(영역별) 변환 규칙 — Phase 2 신규

* **광원 1 + 피사체 N**: 사진 한 장은 한 조명 아래 찍힌 것이므로 `illu_id` 는 단일 유지. 피사체는 영역(region)마다 다르게 매핑.
* **`SS_REGIONS`** = `list[{subject, mask, matrix}]` (session_state 단독 보유 — 마스크는 휘발).
* **합성**: 영역마다 GPT-4o-mini 호출 → 행렬 검증 → `apply_matrix_array(img, M)` 결과를 `out[mask]` 로 덮어쓴다.
  * 시작점은 원본(`out = img.copy()`) — **안 칠한 곳은 자동으로 원본 유지**.
  * 마스크 겹침: **last-write-wins** (루프 후순위가 이긴다).
  * **부분 실패 격리**: 한 영역이 스키마 위반이어도 IDENTITY 로 fallback → 나머지는 정상.
* **DB**: 마스크는 절대 저장하지 않는다. 공유물 = 결과 이미지 + 5색 + `subject_labels` (JSONB 배열) + 메타.
* **좌표계 동기화**: drawable-canvas 표시 크기 ≠ 이미지 H×W → 마스크는 반드시 `cv2.resize(..., (W,H), INTER_NEAREST)` 로 리사이즈. boolean 마스크에 선형 보간을 쓰면 경계가 회색으로 뭉개진다.

---

## 3. DB 스키마 요약

| 테이블                  | 목적                          | 핵심 컬럼                                                                         |
| ----------------------- | ----------------------------- | --------------------------------------------------------------------------------- |
| `standard_illuminants`  | 광원 마스터                   | `name`, `spd_data (JSONB)`                                                        |
| `reflectance_library`   | RAG 반사율 라이브러리         | `category`, `label`, `spectral_values (JSONB)`, `is_uv_active`                    |
| `profiles`              | 사용자 (Supabase Auth 연동)   | `id`, `nickname`                                                                  |
| `projects`              | 시뮬레이션 결과 + 메타        | `owner_id`, `parent_id (self FK)`, `subject_id (FK, 대표)`, `subject_labels (JSONB[])`, `illu_id (FK)`, `matrix_json`, `status`, `scrap_count`, `is_anonymous`, `is_palette_public` |
| `palettes`              | 대표 5색 HEX                  | `project_id (FK CASCADE)`, `base_colors (JSONB)`                                  |

> 📂 **스키마 변경은 Supabase Dashboard → SQL Editor 에서 직접 적용**합니다.
> 본 저장소는 마이그레이션 파일을 별도로 보관하지 않으며,
> 클라이언트 코드는 변경 후의 스키마 (위 표) 를 가정하고 동작합니다.

> ❗ `palettes` 는 `projects` 에 대해 `ON DELETE CASCADE` 이므로 별도 정리 불필요.
> ❗ `projects.parent_id` 는 셀프 FK — 수정 스냅샷 & 스크랩 복제 시 부모를 가리킨다.

---

## 4. 비즈니스 규칙 요약

* **익명 공유**: `projects.is_anonymous = TRUE` → `anon_gallery_view` 에서 노출.
* **스크랩 (인원수만 추적)**: 원본 row 의 `scrap_count += 1` 만 수행하며, 누가 스크랩했는지는 별도 저장하지 않는다.
* **조건부 삭제**: 다음 중 하나라도 참이면 즉시 삭제 불가 → `status='Pending_Deletion'` 전이.
  * `scrap_count >= 1`
  * 공유 후 30일 미경과
* **삭제 예정 처리**: 모달(`show_project_modal`) 내부에서 자동으로 배너를 띄우고 스크랩 버튼을 비활성화한다.
* **RLS**: 시연 기간에는 `USING (true)` 로 열려 있다. 시연 종료 후 `auth.uid()` 기반으로 다시 잠근다.

---

## 5. 휘발성 vs 영속 데이터

| 데이터                          | 저장 위치                       | 비고                          |
| ------------------------------- | ------------------------------- | ----------------------------- |
| 1024px 원본 이미지              | Supabase Storage `/originals/`  | URL 만 `projects.origin_path` |
| 변환 결과 이미지                | Supabase Storage `/results/`    | URL 만 `projects.result_path` |
| 3×4 변환 매트릭스               | `projects.matrix_json`          | GPT-4o 응답                   |
| 대표 5색 HEX                    | `palettes.base_colors`          | JSONB                         |
| **5×10 명채도 타일 (50 swatch)** | **`st.session_state` 만**       | **DB 미저장 — OKLCH 변주로 매번 재계산** |
| RAG 컨텍스트 (분광 / SPD)       | `reflectance_library` + `standard_illuminants` | GPT-4o-mini 에 **이미지 없이** JSON 만 전달 |
