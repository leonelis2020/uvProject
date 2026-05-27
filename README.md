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

| PHASE | 책임                              | 핵심 함수 (위치)                                                   | 외부 의존성                       |
| ----- | --------------------------------- | ------------------------------------------------------------------ | --------------------------------- |
| 0     | Supabase 연결 + 마스터 데이터     | `get_reflectance_data`, `get_illuminant_spd`                       | `st_supabase_connection`          |
| 1     | 이미지 업로드 + 마스킹            | `handle_image_upload`, `render_mask_canvas`                        | `Pillow`, `streamlit-drawable-canvas` |
| 2     | RAG + GPT-4o 멀티모달             | `build_rag_context`, `call_gpt4o_for_matrix`, `validate_matrix_payload` | `openai`                       |
| 3     | 로컬 픽셀 변환 + 5×10 타일        | `apply_matrix_to_masked_region`, `extract_base_colors`, `build_5x10_tile_grid` | `opencv-python`, `scikit-learn` |
| 4     | Storage + DB INSERT (트랜잭션)    | `upload_to_storage`, `save_simulation_result`, `save_snapshot`     | `supabase`                        |
| 5     | 갤러리 / 스크랩 / 조건부 삭제     | `fetch_anon_gallery`, `scrap_project`, `request_project_deletion`, `render_pending_deletion_banner` | `supabase` |

각 PHASE 의 미구현 함수는 `NotImplementedError` 또는 `TODO[PHASE-N]` 주석으로
정확한 삽입 지점을 표시해 두었다.

---

## 3. DB 스키마 요약

| 테이블                  | 목적                          | 핵심 컬럼                                                                         |
| ----------------------- | ----------------------------- | --------------------------------------------------------------------------------- |
| `standard_illuminants`  | 광원 마스터                   | `name`, `spd_data (JSONB)`                                                        |
| `reflectance_library`   | RAG 반사율 라이브러리         | `category`, `label`, `spectral_values (JSONB)`, `is_uv_active`                    |
| `profiles`              | 사용자 (Supabase Auth 연동)   | `id`, `nickname`                                                                  |
| `projects`              | 시뮬레이션 결과 + 메타        | `owner_id`, `parent_id (self FK)`, `illu_id`, `matrix_json`, `status`, `scrap_count`, `is_anonymous` |
| `palettes`              | 대표 5색 HEX                  | `project_id (FK CASCADE)`, `base_colors (JSONB)`                                  |

> ❗ `palettes` 는 `projects` 에 대해 `ON DELETE CASCADE` 이므로 별도 정리 불필요.
> ❗ `projects.parent_id` 는 셀프 FK — 수정 스냅샷 & 스크랩 복제 시 부모를 가리킨다.

---

## 4. 비즈니스 규칙 요약

* **익명 공유**: `projects.is_anonymous = TRUE` → `anon_gallery_view` 에서 노출.
* **스크랩**: 원본 row 의 `scrap_count += 1` + 복제 row(`parent_id=원본`) INSERT.
* **조건부 삭제**: 다음 중 하나라도 참이면 즉시 삭제 불가 → `status='Pending_Deletion'` 전이.
  * `scrap_count >= 1`
  * 공유 후 30일 미경과
* **삭제 예정 배너**: `render_pending_deletion_banner()` 가 자동으로 배너를 띄우고
  스크랩 버튼을 비활성화한다.

---

## 5. 휘발성 vs 영속 데이터

| 데이터                          | 저장 위치                       | 비고                          |
| ------------------------------- | ------------------------------- | ----------------------------- |
| 1024px 원본 이미지              | Supabase Storage `/originals/`  | URL 만 `projects.origin_path` |
| 변환 결과 이미지                | Supabase Storage `/results/`    | URL 만 `projects.result_path` |
| 3×4 변환 매트릭스               | `projects.matrix_json`          | GPT-4o 응답                   |
| 대표 5색 HEX                    | `palettes.base_colors`          | JSONB                         |
| **5×10 명채도 타일 (50 swatch)** | **`st.session_state` 만**       | **DB 미저장 — 실시간 재계산** |
