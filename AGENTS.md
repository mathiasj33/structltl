# jaxltl Agent Instructions

- Run Python through pixi in this workspace. Use `pixi run python ...` for scripts, `pixi run pytest` for tests, and `pixi run ruff check ...` / `pixi run ruff format ...` for linting and formatting.
- The project targets Python 3.12 only; keep changes compatible with the version pinned in [pyproject.toml](pyproject.toml).
- Treat [src/jaxltl/](src/jaxltl) as the main library code, [scripts/](scripts) as runnable utilities, and [test/](test) as the test suite.