# xLLM Coding Agent Instructions

## Code Style Guide

* Before editing, creating, refactoring, or reviewing any file under `xllm/`, you **MUST** read the shared rules in [custom-code-style.md](.agents/skills/code-review/references/custom-code-style.md) and the guide for each language you touch: [C++](.agents/skills/code-review/references/custom-code-style-cpp.md) or [Python](.agents/skills/code-review/references/custom-code-style-python.md).
* The files above are **required instruction files**, not optional references. Do not skip reading them.
* After editing any C++ file, run clang-format 20.1.6 with the repo `.clang-format` on every touched file before finishing. Do not leave formatting for the user to remind you.
* Apply the rules in the shared guide and the matching language guide to **both code generation and code review**.
* Follow DDD (Domain Driven Design) principles, and keep the codebase clean and maintainable.
* If the shared guide or the matching language guide specifies a rule, that rule takes precedence over the Google C++/Python Style Guide.
* Use the Google C++/Python Style Guide only for cases not specified in the shared guide or the matching language guide.

## Review Instructions

* For code review tasks, you **MUST** first read [code-review/SKILL.md](.agents/skills/code-review/SKILL.md).
* Then read the shared rules in [custom-code-style.md](.agents/skills/code-review/references/custom-code-style.md) and the matching language guide(s) ([C++](.agents/skills/code-review/references/custom-code-style-cpp.md), [Python](.agents/skills/code-review/references/custom-code-style-python.md)) and apply them during the review.
* Review code changes for quality, security, performance, correctness, and maintainability following the project-specific standards.
* Review code changes for DDD (Domain Driven Design) principles, and keep the codebase clean and maintainable.
* Use the review workflow, checklist, severity rules, and output format defined in [code-review/SKILL.md](.agents/skills/code-review/SKILL.md).
* Apply the Google C++/Python Style Guide only when the project-specific style guide does not define the rule.
* Focus the review on the requested diff or changed files. Do not comment on unrelated code.