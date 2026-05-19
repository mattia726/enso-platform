# Top-level Makefile for the enso-cell-count product.
#
# All commands are POSIX-shell compatible and assume the repository root
# is the current working directory.

VENV_PY ?= $(if $(wildcard backend/.venv/Scripts/python.exe),backend/.venv/Scripts/python.exe,backend/.venv/bin/python)
PIP_ARGS ?= --no-cache-dir

.PHONY: help backend-install backend-test backend-lint \
        macrodissection-artifacts macrodissection-serve \
        frontend-install frontend-test frontend-build frontend-dev \
        visual-check clean-tests

help:
	@echo "Available targets:"
	@echo "  backend-install              create backend/.venv and install all backend deps"
	@echo "  backend-test                 run the backend pytest suite (incl. macrodissection)"
	@echo "  backend-lint                 run ruff over the backend package"
	@echo "  macrodissection-artifacts    rebuild per-case tile JSON + grid bins from PNG masks"
	@echo "  macrodissection-serve        start the FastAPI server (port 8000)"
	@echo "  frontend-install             npm install in frontend/"
	@echo "  frontend-test                run the vitest TS math suite"
	@echo "  frontend-build               next build (static export)"
	@echo "  frontend-dev                 next dev (live reload on port 3000)"
	@echo "  visual-check                 Playwright capture of the 8 reference screenshots"

backend-install:
	cd backend && python -m venv .venv
	$(VENV_PY) -m pip install $(PIP_ARGS) -U pip setuptools wheel
	$(VENV_PY) -m pip install $(PIP_ARGS) -r backend/requirements.txt
	$(VENV_PY) -m pip install $(PIP_ARGS) -e backend
	$(VENV_PY) -m pip install $(PIP_ARGS) Pillow shapely

backend-test:
	$(VENV_PY) -m pytest backend/tests -q

backend-lint:
	$(VENV_PY) -m ruff check backend/src/enso_purity backend/tests

macrodissection-artifacts:
	$(VENV_PY) -m enso_purity.macrodissection.build_artifacts \
		--cases-dir frontend/public/cases \
		--gallery frontend/gallery/gallery_summary.csv

macrodissection-serve:
	ENSO_CASES_DIR=frontend/public/cases ENSO_ROIS_DIR=backend/.runtime/rois \
		$(VENV_PY) -m uvicorn enso_purity.api.main:app --host 0.0.0.0 --port 8000

frontend-install:
	cd frontend && npm install

frontend-test:
	cd frontend && npm test --silent

frontend-build:
	cd frontend && npm run build

frontend-dev:
	cd frontend && npm run dev

visual-check:
	cd frontend && npm run build && \
		(npx serve out -p 3010 & echo $$! > .serve.pid && sleep 4) && \
		ENSO_WORKBENCH_URL=http://localhost:3010/macrodissection \
			ENSO_SCREENSHOT_DIR=$$(pwd)/docs/screenshots \
			npx playwright test tests/visual/macrodissection.spec.ts ; \
		test -f .serve.pid && kill `cat .serve.pid` 2>/dev/null ; rm -f .serve.pid

clean-tests:
	rm -rf frontend/test-results frontend/test-screenshots
