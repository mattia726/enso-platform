# Prompt: Scaffold + repo hygiene

You are working in this repo. Create / refine the repo structure and tooling:

Requirements:
- Keep monorepo structure: `backend/` and `frontend/` (frontend may be generated later).
- Add a `Makefile` (or `justfile`) with: `setup`, `test`, `lint`, `format`, `run-api`.
- Ensure `pytest -q` passes.
- Add `.gitignore` for Python, Node, and outputs.
- Add `pre-commit` hooks (optional) only if low-friction.

Deliverables:
- Minimal, clean scaffolding
- Clear README instructions
- No dead code
