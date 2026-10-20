# Custom Code Style: Python

Python-specific rules for xllm. Shared rules that apply to every language
(copyright headers and the online serving failure/fallback policy) are in
[custom-code-style.md](custom-code-style.md).

---

## 1. Naming Conventions

| Element          | Style                  | Example                              |
|------------------|------------------------|--------------------------------------|
| Module / file    | `snake_case`           | `model_loader.py`                    |
| Class            | `PascalCase`           | `TokenizerConfig`                    |
| Function         | `snake_case`           | `load_model`                         |
| Variable         | `snake_case`           | `batch_size`                         |
| Constant         | `ALL_CAPS`             | `MAX_SEQ_LEN`                        |
| Private member   | `_leading_underscore`  | `_internal_state`                    |

---

## 2. Scoping & Visibility

- **File-local helper functions** (not part of the public API) must be prefixed with `_`.
- **Non-public member functions** of a class must be prefixed with `_`.

```python
def _validate_config(config: dict) -> bool:
    ...

class ModelLoader:
    def load(self, path: str) -> Model:
        self._check_path(path)
        ...

    def _check_path(self, path: str) -> None:
        ...
```

---

## 3. Python-Specific Rules

- **Type annotations are required** on all function signatures (parameters and return types). Use `typing` module types where needed.

```python
# Good
def load_model(path: str, device: str = "cuda") -> nn.Module:
    ...

# Bad
def load_model(path, device="cuda"):
    ...
```

- **Private helpers**: prefix with `_` (see Section 2).
- **Prefer absolute imports** rooted at the top-level package for project code.
  - A single-dot relative import (`from .module import Name`) is acceptable for a sibling module in the same package, especially for package-local implementation details and `__init__.py` re-exports.
  - Avoid parent-relative imports (`from ..module import Name`) and deeper relative imports. Use an absolute import from `xllm` instead so the dependency remains clear when code moves between packages.

```python
# Preferred across packages
from xllm.python.attention.backend import AttentionBackend

# Acceptable within the same package
from .backend import AttentionBackend

# Avoid
from ..attention.backend import AttentionBackend
```

---

## 4. Python Logging

All Python diagnostic output **MUST** go through the shared logger at `scripts/logger.py`. Do not use `print()` for logging, do not call `logging.basicConfig`, and do not create per-module loggers via `logging.getLogger(...)`.

- **Import:** `from scripts.logger import logger`. For standalone `tools/*.py`, prepend repo root to `sys.path` first: `sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))`.
- **Levels:** `debug` (investigation only), `info` (normal progress), `warning` (recoverable anomaly / fallback), `error` (failure, including pre-`exit(1)`), `fatal` (process-fatal). Inside `except`, prefer `logger.exception(...)` to keep the traceback.
- **No level prefixes in messages.** The formatter already emits `I/W/E/F/D` plus `file:line`. Status emojis (`✅ ❌ ⚠️ 🔨 🚀 ℹ️`) are allowed.
- **`print()` is only for program output**, not diagnostics: model results in `examples/`, CLI stdout meant to be piped, or verbatim streaming of subprocess output (e.g. `sys.stdout.write(line)` for ctest).
- **Do not reconfigure the shared logger** from library code: no `setLevel`, no extra handlers, no `propagate` changes.

```python
# Good
from scripts.logger import logger
logger.info(f"compiled kernel: {name}")
logger.warning("Skip reference check: NPU is not available")
try:
    do_work(path)
except OSError:
    logger.exception(f"failed to process {path}")
```

```python
# Bad
import logging
logger = logging.getLogger(__name__)            # re-implementing a logger
print(f"[INFO] build start: device={device}")   # diagnostic via print
logger.warning("[WARN] missing --device")       # duplicated level prefix
logger.info("build failed, exiting")            # wrong level for a failure
```
