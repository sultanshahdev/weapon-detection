# Development instructions

## Project identity

Real-Time Weapon Detection and AI-Powered Threat Intelligence System is a local academic
prototype for identifying visible firearms in video and presenting evidence for human
review. The 48-hour MVP prioritizes a repeatable prerecorded-video demonstration.

## Mandatory context loading

Read `CONTEXT.md`, `PLANS.md`, `TASKS.md`, and `docs/ARCHITECTURE.md` before changing the
project. Read the dataset, evaluation, and demo documents as relevant.

## Development priorities

In order, favor working functionality over speculative features, correctness over visual
complexity, reusable modules over duplicated code, measured results over unsupported
claims, explicit errors over silent failures, and simple architecture over unnecessary
abstractions.

## Coding standards

Use descriptive names, small functions, useful type hints, explicit errors, and clear
module boundaries. Keep UI and business logic separate. Avoid circular imports, hardcoded
credentials, hardcoded dataset class indices, and unnecessary global mutable state. Use
managed temporary files with predictable cleanup, including on failure.

## Testing standards

Unit-test nontrivial logic with mocked inference. Ordinary tests must run on CPU with
no footage, checkpoints, or network. Verify Streamlit startup and run Ruff. Report
untested behavior explicitly.

## Dependency management

Use Python 3.11, uv, the existing pyproject.toml, and uv.lock. Keep the MVP Python-only
and use standard-library features for SQLite, CSV, and temporary files.

## Dataset and model rules

Inspect dataset licenses and validate annotations before use. Split by source before
training. Use trusted local checkpoints and keep CPU inference as the default. Inspect
actual checkpoint class names at runtime. Report measured detections
and evaluation results only. Support accuracy claims with reproducible evidence.

## Security and privacy

Keep credentials and private surveillance footage out of Git. Store sensitive evidence
only in configured local directories. Process video locally, require human review of
incidents, and document prototype limitations. Keep web-facing paths scoped to
trusted local inputs and configured output directories. Verify both positive and
negative clips when changing temporal confirmation or persistence.

## Session completion protocol

1. Run relevant tests and Ruff.
2. Update TASKS.md, PLANS.md when milestones change, and CONTEXT.md when decisions change.
3. Update architecture documentation when module relationships change.
4. Report changed files, commands, test outcomes, remaining blockers, and the next concrete action.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

When the user types `/graphify`, use the installed graphify skill or instructions before doing anything else.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- Dirty graphify-out/ files are expected after hooks or incremental updates; dirty graph files are not a reason to skip graphify. Only skip graphify if the task is about stale or incorrect graph output, or the user explicitly says not to use it.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
