---
name: Part2 MLE Bootcamp — Context & Tech Stack
description: Technical context and stack of the /part2 MLE bootcamp directory — relevant background for upcoming SayFit capstone work
type: project
---
# Part2 — MLE Bootcamp (Machine Learning Engineering)

**Path:** `/Users/kevin/Documents/bootcamp_ai/part2`
**Context:** Kevin's MLE bootcamp curriculum, organized by week. Structured learning program building toward the SayFit capstone project.

---

## Overall Structure

```
part2/
├── week1/  — Python packaging, unit/integration testing, refactoring & OOP
├── week2/  — FastAPI, Docker, Kubernetes, refactoring projects
├── week3/  — Data pipelines (Prefect), batch/stream processing, dbt, DuckDB
├── week4/  — ML basics, MLflow, FastAPI ML deployment, testing for ML
├── week5/  — MLOps: DVC + GitHub Actions CI/CD; monitoring: Evidently, Prometheus, Grafana
```

---

## Week-by-Week Summary

### Week 1 — Engineering Foundations
- Jupyter notebooks → Python modules refactoring
- Python packaging (`pyproject.toml`)
- Unit and integration testing
- OOP refactoring patterns
- **Tech:** Python 3.x, pytest, pyproject.toml

### Week 2 — APIs & Containerization
- **FastAPI** for ML-serving REST APIs
- **Docker** (Dockerfile, docker-compose)
- **Kubernetes** (deployment.yaml, service.yaml basics)
- Refactoring project with service/database/model/schema layers
- **Tech:** FastAPI, Docker, Docker Compose, Kubernetes

### Week 3 — Data Pipelines
- **Prefect** workflow orchestration (ETL + ELT flows)
- **dbt** for data transformations (DuckDB + Postgres targets)
- Batch and stream processing concepts
- **Tech:** Prefect, dbt, DuckDB, PostgreSQL, Docker Compose, pandas

### Week 4 — ML Deployment
- ML model basics and evaluation
- **MLflow** for experiment tracking and model registry
- FastAPI as ML webservice layer (`predict.py`, `data_model.py`)
- Testing for ML (refactoring, integration testing with pytest)
- Deployment project: end-to-end ML API in Docker
- **Tech:** MLflow, FastAPI, sklearn, pytest, Docker

### Week 5 — MLOps & Monitoring (active week)
- **DVC** (Data Version Control) — `dvc.yaml` pipeline stages for reproducible training
- **GitHub Actions CI/CD** — automated training pipelines via `.github/workflows/train.yaml`
- **Evidently** — ML monitoring (data drift, batch predictions)
- **Prometheus + Grafana** — real-time service and ML monitoring
- Active project: `mle-mlops-project/` — DVC pipeline + GitHub Actions integration
- **Tech:** DVC, GitHub Actions, Evidently, Prometheus, Grafana, MLflow, Docker Compose, Prefect

---

## Active Project: mle-mlops-project (week5)

**Path:** `/Users/kevin/Documents/bootcamp_ai/part2/week5/mle-mlops-project`

| File | Purpose |
|------|---------|
| `dvc.yaml` | DVC pipeline stages (data loading → preprocessing → training) |
| `src/data_loading.py` | Data loading stage |
| `src/preprocessing.py` | Feature engineering |
| `src/train.py` | Model training with MLflow tracking |
| `src/config.py` | Config/paths |
| `.github/workflows/train.yaml` | GitHub Actions CI for automated training |
| `plan.md` | Implementation plan |
| `CLAUDE.md` | Claude Code project instructions |

Data: NYC green taxi trip data (parquet), tracked via DVC.

---

## Full Tech Stack (recurring across part2)

| Category | Tools |
|----------|-------|
| Language | Python 3.x |
| APIs | FastAPI |
| Containers | Docker, Docker Compose |
| Orchestration | Kubernetes (basics), Prefect |
| Data | pandas, DuckDB, PostgreSQL |
| Transforms | dbt |
| ML tracking | MLflow |
| Data versioning | DVC |
| CI/CD | GitHub Actions |
| Monitoring | Evidently, Prometheus, Grafana |
| Testing | pytest |
| Packaging | pyproject.toml |

**Why:** This is the production ML engineering stack Kevin learned in the bootcamp and should prefer for capstone work.
**How to apply:** For the upcoming capstone (SayFit or otherwise), prefer DVC for data versioning, FastAPI for serving, Docker for deployment, MLflow for experiment tracking, GitHub Actions for CI/CD — these are all familiar to Kevin.
