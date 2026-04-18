# Project Instructions for AI Agents

This file provides instructions and context for AI coding agents working on this project.

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:ca08a54f -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files
- After closing a beads issue, run `bd ready` to find the next available issue and continue working on it

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd dolt push
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
- **EXCEPTION: No remote origin** - If `git remote get-url origin` fails (no origin configured), skip the push step but WARN the user: "No git remote origin is configured. You should set one up with `git remote add origin <url>` to avoid losing work."
<!-- END BEADS INTEGRATION -->


## Build & Test

Project layout: repo root contains `egg-toolbox/` (the Python project with pyproject.toml, tests/, and the `egg_toolbox/` package), plus `docs/`, `.beads/`, `.venv/`.

```bash
# Create/activate venv (at repo root)
python3 -m venv .venv
source .venv/bin/activate

# Install with dev deps
pip install -e "./egg-toolbox[dev]"

# Run tests
.venv/bin/python -m pytest egg-toolbox/tests/

# Quick import check
.venv/bin/python -c "from egg_toolbox.types import *; print('OK')"

# Start the server
.venv/bin/python -m egg_toolbox path/to/model.gguf --backend tinygrad
```

## Architecture Overview

See `egg-toolbox/context.md` for full architecture. Key flow:
- API Request -> Orchestrator -> ChatTemplate.render() -> Backend.generate_tokens() -> StreamingParser -> SemanticEvent stream -> API SSE projection
- Universal IR: `SemanticEvent` -- both OpenAI and Anthropic APIs are lossless projections

## Conventions & Patterns

- Pure Python, no Rust/C extensions
- Frozen dataclasses (not Pydantic) for all data types
- Starlette (not FastAPI) for HTTP
- Two backend ABCs: StepBackend (token-by-token) vs ConstraintBackend (text chunks)
- Per-format state machine parsers (not regex) for streaming
- Format detection at model load (not per-request)
- Design docs: `docs/egg-toolbox-plan.md`, `docs/egg-toolbox-architecture.md`
- Session continuation: `docs/egg-toolbox-new-session.md`
