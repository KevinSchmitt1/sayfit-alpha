## Integration of new tests

- tests/: add new test files here — they get pulled automatically by gh-actions (no changes to ci.yml needed)
- exception: new CI steps like docker build or postgres service require editing ci.yml directly

## Relevant skills from everything-claude-code (plugin, skills live in .claude/skills/)

- /python-testing: when you write more tests (parametrize, fixture patterns, coverage config)
- /tdd-workflow: for writing tests before new features in Week 1
- /github-ops: after pushing, use this to enable branch protection on main (requires green CI before any merge — this is what actually enforces the safety net for your teammates)
- /verification-loop: run locally before every PR to catch issues before CI does

## Secrets

- API-keys (like openai or groq): are saved as secrets in github-actions (Settings → Secrets → Actions)
- DB_PASSWORD: add in Week 2 when Postgres is introduced
- LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY: add in Week 3

## Week 1: minimum integration of CI

- test_smoke.py: tells the extractor not to use llm. The text parser gets used instead (no-llm-solution). At this point we don't care about the quality, but that the pipeline is running from beginning to end

- conftest.py: creates a hardcoded list, that prevents the retriever (step2) to make FAISS vector search. The `retrieve()` function gets substituted by a fake-candidate

- ruff check: catches syntax errors, undefined names, unused imports (ignores the voice_input, since PortAudio does not exist on CI server)

## Week 2: API + Docker (additions to ci.yml)

- add `docker build` step: verifies the image builds without errors on every push
- add `services: postgres` block: spins up a real Postgres container so API integration tests can run against a real database
- add test_api.py: tests FastAPI endpoints using TestClient — checks status codes and response shapes (still uses conftest mocks, picked up automatically)

## Week 3: Langfuse + Prometheus (additions to ci.yml)

- add `LANGFUSE_ENABLED=false` env var: Langfuse SDK runs in no-op mode in CI — the full stack (clickhouse, redis, worker) is never started in CI
- add prometheus check step: curl /metrics after docker run to verify the endpoint is exposed
- note: model quality monitoring happens through real pipeline runs via Langfuse, not through pytest — CI only checks that the instrumentation is wired up correctly
