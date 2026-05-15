# SayFit — Data Engineering Scope

## Overview

The goal of the Data Engineering part is to transform the current notebook-based food data preparation workflow into a reproducible, modular, and testable data pipeline.

The pipeline will focus on preparing the nutrition datasets used by the SayFit retrieval system (`combined_final.csv`).

---

# 1. Workflow Orchestration with Prefect

## Goal

Use Prefect to orchestrate and monitor the complete data preparation workflow.

## Planned Pipeline

```text
Load USDA Data
    ↓
Load OpenFoodFacts Data
    ↓
Clean & Normalize Data
    ↓
Deduplicate USDA Entries
    ↓
Merge Data Sources
    ↓
Validate Final Dataset
    ↓
Export combined_final.csv
```

# 2. Data Storage Layer

## Goal

Introduce a structured storage layer for raw, cleaned, and transformed datasets.

Instead of relying only on CSV files and notebooks, datasets will be stored in database tables to improve organization and reproducibility.

## Planned Tools

Currently evaluating:

- DuckDB
- PostgreSQL

DuckDB is currently preferred for local analytical workflows and easier setup.

## Planned Structure

```text
raw_usda
raw_openfoodfacts
clean_usda
clean_openfoodfacts
food_items_final
```

# 3. Data Transformations with dbt

## Goal

Use dbt to modularize and manage SQL-based data transformations.

The objective is to replace large notebook-based transformations with smaller, testable, and reproducible SQL models.

## Planned dbt Structure

```text
staging/
intermediate/
marts/
```


# 4. Data Versioning with DVC

## Goal

Introduce reproducible dataset versioning for raw and processed nutrition datasets.

The objective is to track changes to datasets over time and ensure that pipeline outputs can always be reproduced.

## Planned Tool

- DVC (Data Version Control)

## Planned Features

- Version large datasets
- Track processed dataset outputs
- Reproduce previous dataset versions
- Compare dataset changes over time
- Improve collaboration between team members

## Planned Usage

- Track raw nutrition datasets
- Track cleaned datasets
- Track final pipeline outputs
- Connect dataset versions to pipeline runs

## Why

The nutrition datasets are central to the retrieval quality of SayFit.

Data versioning ensures that:
- dataset changes remain traceable
- retrieval behavior can be reproduced
- previous dataset states can be restored
- pipeline outputs remain consistent across environments

# 5. Pipeline Monitoring & Logging

## Goal

Introduce monitoring and logging for pipeline execution and task-level visibility.

The objective is to improve transparency, debugging, and reproducibility of the data pipeline.

## Planned Tool

- Prefect

## Planned Monitoring Features

- task execution tracking
- runtime logging
- pipeline state monitoring
- error visibility
- row count logging
- execution duration tracking

## Planned Logging Examples

```text
Rows loaded from USDA
Rows loaded from OpenFoodFacts
Rows removed during cleaning
Number of duplicates removed
Final dataset size
Pipeline runtime
```