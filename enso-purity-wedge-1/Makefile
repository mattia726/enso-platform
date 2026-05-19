@@
 VENV_PY = $(if $(wildcard backend/.venv/Scripts/python.exe),backend/.venv/Scripts/python.exe,backend/.venv/bin/python)
+PIP_ARGS ?= --no-cache-dir
@@
 backend-install:
 	cd backend && python -m venv .venv
-	$(VENV_PY) -m pip install -U pip setuptools wheel
-	$(VENV_PY) -m pip install -r backend/requirements.txt
+	$(VENV_PY) -m pip install $(PIP_ARGS) -U pip setuptools wheel
+	$(VENV_PY) -m pip install $(PIP_ARGS) -r backend/requirements.txt
+	$(VENV_PY) -m pip install $(PIP_ARGS) -e backend