-- ============================================================
-- 2026-06-18 마이그레이션
--   * projects.subject_id  (FK -> reflectance_library) 추가
--   * projects.is_palette_public BOOLEAN DEFAULT TRUE 추가
--   * 조회 성능을 위한 인덱스 추가
-- ============================================================

-- 1) 피사체 참조 컬럼
ALTER TABLE public.projects
    ADD COLUMN IF NOT EXISTS subject_id UUID REFERENCES public.reflectance_library(id);

COMMENT ON COLUMN public.projects.subject_id IS
    '이 프로젝트에서 사용한 피사체(reflectance_library 참조). 광원(illu_id)과 함께 변환 행렬의 입력 근거.';

-- 2) 팔레트 공개 여부
ALTER TABLE public.projects
    ADD COLUMN IF NOT EXISTS is_palette_public BOOLEAN DEFAULT TRUE;

COMMENT ON COLUMN public.projects.is_palette_public IS
    '팔레트 5색 + 5x10 그리드를 갤러리에서 공개할지 여부. FALSE 이면 본인만 조회 가능.';

-- 3) 조회 성능 인덱스
CREATE INDEX IF NOT EXISTS idx_projects_subject_id ON public.projects(subject_id);
CREATE INDEX IF NOT EXISTS idx_projects_illu_id    ON public.projects(illu_id);
CREATE INDEX IF NOT EXISTS idx_projects_owner_id   ON public.projects(owner_id);

-- 4) (옵션) 익명 갤러리 뷰 — anon_gallery_view
--     좌측 사이드바 "남이 공유한 미리보기" 가 1차로 참조한다. 존재하지 않으면
--     app.py 의 fetch_others_shared_projects() 가 projects 테이블로 fallback 한다.
CREATE OR REPLACE VIEW public.anon_gallery_view AS
SELECT
    p.id,
    p.owner_id,
    p.subject_id,
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

-- 5) FK 무결성 점검 (참고용 SELECT — 실행해도 부작용 없음)
-- SELECT tc.constraint_name, tc.constraint_type, kcu.column_name,
--        ccu.table_name AS references_table, ccu.column_name AS references_column
-- FROM information_schema.table_constraints tc
-- JOIN information_schema.key_column_usage kcu
--   ON tc.constraint_name = kcu.constraint_name
-- LEFT JOIN information_schema.constraint_column_usage ccu
--   ON tc.constraint_name = ccu.constraint_name
-- WHERE tc.table_name = 'projects' AND tc.constraint_type = 'FOREIGN KEY';
