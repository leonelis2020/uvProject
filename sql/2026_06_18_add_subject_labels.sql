-- ============================================================
-- 2026-06-18 마이그레이션 (Phase 2 — 멀티리전 피사체)
--   * projects.subject_labels JSONB 추가 (영역별 피사체 라벨 배열)
--   * anon_gallery_view 재정의 (신규 컬럼 포함)
-- ============================================================

-- 1) 영역별 피사체 라벨 배열
--    예: ["조류 강반사", "조류 약반사", "알껍데기"]
ALTER TABLE public.projects
    ADD COLUMN IF NOT EXISTS subject_labels JSONB;

COMMENT ON COLUMN public.projects.subject_labels IS
    '멀티리전 변환에서 각 영역에 매핑된 피사체 라벨 배열 (JSONB). '
    '단일 subject_id 와 함께 사용되며, subject_id 는 대표(첫) 피사체.';

-- (참고) 조인 테이블(project_subjects)은 만들지 않는다.
-- 재현/편집을 지원하지 않으므로 과정규화는 불필요.

-- 2) 익명 갤러리 뷰 — 신규 컬럼을 포함하도록 재정의
CREATE OR REPLACE VIEW public.anon_gallery_view AS
SELECT
    p.id,
    p.owner_id,
    p.subject_id,
    p.subject_labels,
    p.illu_id,
    p.matrix_json,
    p.origin_path,
    p.result_path,
    p.status,
    p.is_anonymous,
    p.is_palette_public,
    p.scrap_count,
    p.created_at
FROM public.projects p
WHERE p.is_anonymous = TRUE
  AND p.status = 'Active';
