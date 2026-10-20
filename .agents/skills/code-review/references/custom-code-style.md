# Custom Code Style

Project-specific coding style for xllm. The reviewer **MUST** enforce these style.

This guide is split by language. Read this file for the rules shared by all
languages, then read the guide for each language you are editing or reviewing:

| Language | Guide                                                      |
|----------|------------------------------------------------------------|
| C++      | [custom-code-style-cpp.md](custom-code-style-cpp.md)       |
| Python   | [custom-code-style-python.md](custom-code-style-python.md) |

---

## 1. File & Header Rules

- **Copyright header required** on all new files. Use the correct year matching the file creation date.

```cpp
/* Copyright 2025 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
...
==============================================================================*/
```

---

## 2. Online Serving Failures and Fallbacks

- **Prefer the let-it-crash principle for unexpected failures in online execution paths.** Model execution, graph capture/replay, device operations, and runtime state updates should normally let unexpected exceptions propagate instead of catching them and continuing the request.
- **Do not introduce `try/catch` or `try/except` casually in online execution paths.** In particular, do not swallow an exception, retry the operation, fall back to another execution path, or keep serving from state that may already be partially modified.
- **Do not introduce fallback code casually.** A fallback is appropriate only when it is an explicitly supported path selected before execution through observable conditions, and both paths have defined and tested semantics. It must not be used to mask a correctness issue, unsupported state, or failed operation.
- Express expected unsupported conditions through explicit validation, `Status`, or return values before mutating execution state. Do not use exceptions as normal capability detection.
- Exception handling remains valid at required process, RPC, C ABI, untrusted-input, or third-party API boundaries. Such handlers must catch the narrowest practical exception, preserve the failure signal, and must not pretend that a partially failed online operation succeeded.

```cpp
// Good: choose a supported path before execution starts.
if (!graph_supported(params)) {
  return forward_eager(model, params);
}
return forward_graph(model, params);

// Bad: graph execution may have modified runtime state before failing.
try {
  return forward_graph(model, params);
} catch (...) {
  return forward_eager(model, params);
}
```
