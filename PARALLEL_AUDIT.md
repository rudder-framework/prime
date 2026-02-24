# Parallel / Concurrency Audit — Prime Codebase

**Generated:** 2026-02-24
**Scope:** `/Users/jasonrudder/prime` — all `.py`, `.sql`, `.md`, `.yaml`, `.toml`, `.cfg`, `.ini`, `.env` files
**Method:** Grep search for 19 parallel/concurrency patterns (listed at bottom)

---

## All Matches

### 1. `ProcessPoolExecutor`

| File | Line | Snippet | Category |
|------|------|---------|----------|
| `prime/ingest/typology_raw.py` | 21 | `from concurrent.futures import ProcessPoolExecutor, as_completed` | IMPORT |
| `prime/ingest/typology_raw.py` | 1245 | `with ProcessPoolExecutor(max_workers=workers) as pool:` | FUNCTION |
| `prime/sql/typology/runner.py` | 355 | `Called in parallel via ProcessPoolExecutor.` | DOCS (comment — inaccurate: actual code uses ThreadPoolExecutor) |

### 2. `ThreadPoolExecutor`

| File | Line | Snippet | Category |
|------|------|---------|----------|
| `prime/sql/typology/runner.py` | 274 | `from concurrent.futures import ThreadPoolExecutor, as_completed` | IMPORT |
| `prime/sql/typology/runner.py` | 307 | `with ThreadPoolExecutor(max_workers=n_workers) as executor:` | FUNCTION |

### 3. `multiprocessing`

No matches.

### 4. `concurrent.futures`

| File | Line | Snippet | Category |
|------|------|---------|----------|
| `prime/ingest/typology_raw.py` | 21 | `from concurrent.futures import ProcessPoolExecutor, as_completed` | IMPORT |
| `prime/sql/typology/runner.py` | 274 | `from concurrent.futures import ThreadPoolExecutor, as_completed` | IMPORT |

### 5. `as_completed`

| File | Line | Snippet | Category |
|------|------|---------|----------|
| `prime/ingest/typology_raw.py` | 21 | `from concurrent.futures import ProcessPoolExecutor, as_completed` | IMPORT |
| `prime/ingest/typology_raw.py` | 1257 | `for fut in as_completed(futures):` | FUNCTION |
| `prime/sql/typology/runner.py` | 274 | `from concurrent.futures import ThreadPoolExecutor, as_completed` | IMPORT |
| `prime/sql/typology/runner.py` | 315 | `for future in as_completed(futures):` | FUNCTION |

### 6. `_parallel_map`

No matches.

### 7. `parallel_map`

No matches.

### 8. `max_workers`

| File | Line | Snippet | Category |
|------|------|---------|----------|
| `prime/ingest/typology_raw.py` | 1245 | `with ProcessPoolExecutor(max_workers=workers) as pool:` | FUNCTION |
| `prime/sql/typology/runner.py` | 307 | `with ThreadPoolExecutor(max_workers=n_workers) as executor:` | FUNCTION |

### 9. `n_workers`

| File | Line | Snippet | Category |
|------|------|---------|----------|
| `prime/pipeline.py` | 259 | `n_workers=workers or 4,` | FUNCTION (call to `run_sql_typology`) |
| `prime/pipeline.py` | 293 | `n_workers=workers or 4,` | FUNCTION (call to `run_sql_typology` in compare mode) |
| `prime/sql/typology/runner.py` | 61 | `n_workers: int = 4,` | FUNCTION (parameter of `run_sql_typology`) |
| `prime/sql/typology/runner.py` | 82 | `n_workers : int` | DOCS (docstring) |
| `prime/sql/typology/runner.py` | 104 | `con.execute(f"SET threads = {n_workers}")` | CONFIG (DuckDB thread count) |
| `prime/sql/typology/runner.py` | 217 | `n_workers=n_workers,` | FUNCTION (passed to `_compute_expensive_primitives`) |
| `prime/sql/typology/runner.py` | 262 | `n_workers: int = 4,` | FUNCTION (parameter of `_compute_expensive_primitives`) |
| `prime/sql/typology/runner.py` | 307 | `with ThreadPoolExecutor(max_workers=n_workers) as executor:` | FUNCTION |

### 10. `num_workers`

No matches.

### 11. `MANIFOLD_WORKERS`

| File | Line | Snippet | Category |
|------|------|---------|----------|
| `CLAUDE.md` | 532 | `\| MANIFOLD_WORKERS \| Parallel cohorts (Manifold's concern) \| 0 (auto) \|` | DOCS |

### 12. `PRIME_WORKERS`

(Not in original search list but critical — found via `n_workers` search context)

| File | Line | Snippet | Category |
|------|------|---------|----------|
| `CLAUDE.md` | 529 | `\| PRIME_WORKERS \| Parallel signals in typology \| 1 (single-threaded) \|` | DOCS |
| `prime/pipeline.py` | 42 | `PRIME_WORKERS env var or default (4).` | DOCS (docstring) |
| `prime/ingest/typology_raw.py` | 97 | `# Parallel workers — set PRIME_WORKERS=N to override, default = 4` | DOCS (comment) |
| `prime/ingest/typology_raw.py` | 98 | `PRIME_WORKERS = int(os.environ.get("PRIME_WORKERS", "0")) or 4` | CONFIG |
| `prime/ingest/typology_raw.py` | 1206 | `Set PRIME_WORKERS=N for N-way parallel computation, or pass workers= directly.` | DOCS (docstring) |
| `prime/ingest/typology_raw.py` | 1212 | `workers: Number of parallel workers. None = use PRIME_WORKERS env or default (4).` | DOCS (docstring) |
| `prime/ingest/typology_raw.py` | 1217 | `workers = workers if workers is not None else PRIME_WORKERS` | FUNCTION |
| `prime/__main__.py` | 50 | `help='Number of parallel workers for typology (default: PRIME_WORKERS env or 4)')` | CONFIG (CLI argparse) |

### 13. `parallel/`

No matches (no parallel subdirectory references).

### 14. `fork`

No matches.

### 15. `mp_context`

No matches.

### 16. `pool.submit`

| File | Line | Snippet | Category |
|------|------|---------|----------|
| `prime/ingest/typology_raw.py` | 1249 | `fut = pool.submit(` | FUNCTION |

### 17. `pool.map`

No matches.

### 18. `pool.shutdown`

No matches.

### 19. `psutil`

| File | Line | Snippet | Category |
|------|------|---------|----------|
| `pyproject.toml` | 45 | `"psutil>=5.9.0",` | DEPENDENCY |
| `prime/streaming/data_sources.py` | 172 | `import psutil` | IMPORT |
| `prime/streaming/data_sources.py` | 174 | `raise ImportError("psutil required for system metrics: pip install psutil")` | FUNCTION |
| `prime/streaming/data_sources.py` | 177 | `last_disk_read = psutil.disk_io_counters().read_bytes` | FUNCTION |
| `prime/streaming/data_sources.py` | 178 | `last_disk_write = psutil.disk_io_counters().write_bytes` | FUNCTION |
| `prime/streaming/data_sources.py` | 179 | `last_net_sent = psutil.net_io_counters().bytes_sent` | FUNCTION |
| `prime/streaming/data_sources.py` | 180 | `last_net_recv = psutil.net_io_counters().bytes_recv` | FUNCTION |
| `prime/streaming/data_sources.py` | 185 | `cpu_percent = psutil.cpu_percent(interval=None)` | FUNCTION |
| `prime/streaming/data_sources.py` | 186 | `memory_percent = psutil.virtual_memory().percent` | FUNCTION |
| `prime/streaming/data_sources.py` | 189 | `disk_counters = psutil.disk_io_counters()` | FUNCTION |
| `prime/streaming/data_sources.py` | 196 | `net_counters = psutil.net_io_counters()` | FUNCTION |

### 20. `cpu_count`

No matches.

---

## Additional Concurrency Patterns Found

These were not in the original 19 search terms but were discovered during analysis.

### `threading` (thread synchronization)

| File | Line | Snippet | Category |
|------|------|---------|----------|
| `prime/services/job_manager.py` | 30 | `import threading` | IMPORT |
| `prime/services/job_manager.py` | 115 | `self._lock = threading.Lock()` | FUNCTION |
| `prime/services/job_manager.py` | 120 | `self._queue_lock = threading.Lock()` | FUNCTION |
| `prime/services/job_manager.py` | 31 | `from queue import Queue` | IMPORT |

### `executor.submit` (variant of `pool.submit`)

| File | Line | Snippet | Category |
|------|------|---------|----------|
| `prime/sql/typology/runner.py` | 310 | `future = executor.submit(` | FUNCTION |

### `asyncio` (async I/O — websocket server, not computation parallelism)

| File | Line | Snippet | Category |
|------|------|---------|----------|
| `prime/streaming/websocket_server.py` | 10 | `import asyncio` | IMPORT |
| `prime/streaming/websocket_server.py` | 108 | `loop = asyncio.get_event_loop()` | FUNCTION |
| `prime/streaming/websocket_server.py` | 142 | `await asyncio.sleep(0.01)` | FUNCTION |
| `prime/streaming/websocket_server.py` | 156 | `await asyncio.sleep(1)` | FUNCTION |

### `SET threads` (DuckDB internal thread config)

| File | Line | Snippet | Category |
|------|------|---------|----------|
| `prime/sql/typology/runner.py` | 104 | `con.execute(f"SET threads = {n_workers}")` | CONFIG |

### Rayon (Rust-level parallelism in pmtvs dependency — comment only in Prime)

| File | Line | Snippet | Category |
|------|------|---------|----------|
| `packages/dynamics/src/dynamics/ftle.py` | 230 | `# Rust rolling_ftle: Rayon-parallel Jacobian FTLE over all windows` | DOCS (comment) |

---

## Summary

### Total Match Count

- **19 original search terms:** 36 matches (excluding zero-result terms)
- **Additional concurrency patterns:** 9 matches
- **Grand total:** 45 matches across 7 files + 1 doc file

### Files That Would Need Editing to Remove All Parallel Code

| # | File | What to change |
|---|------|---------------|
| 1 | `prime/ingest/typology_raw.py` | Remove `ProcessPoolExecutor` import, `PRIME_WORKERS` constant, parallel branch in `compute_typology_raw()`. Keep only the sequential `else` branch (lines 1268+). |
| 2 | `prime/sql/typology/runner.py` | Remove `ThreadPoolExecutor` import, replace `_compute_expensive_primitives()` parallel loop with sequential iteration. Remove `n_workers` parameter or make it DuckDB-only. Remove `SET threads` DuckDB config or hardcode to 1. |
| 3 | `prime/pipeline.py` | Remove `workers` parameter threading through `run_pipeline()`, `_run_typology_python()`, `_run_typology_sql()`, `_run_typology_compare()`. Remove `n_workers=workers or 4` in calls to `run_sql_typology`. |
| 4 | `prime/__main__.py` | Remove `--workers` CLI argument. |
| 5 | `prime/services/job_manager.py` | Remove `threading` import, replace `threading.Lock()` with no-ops or remove if single-threaded only. Remove `from queue import Queue`. |
| 6 | `prime/streaming/data_sources.py` | Remove `psutil` usage in `SystemMetricsConnector`. (Note: psutil is not parallel code itself — it is a system monitoring library.) |
| 7 | `prime/streaming/websocket_server.py` | Remove `asyncio` usage. (Note: this is async I/O for WebSocket serving, not computation parallelism.) |

**Core parallel files (computation):** Files 1-4 (typology parallelism)
**Thread-safety files:** File 5 (job manager locks)
**System monitoring:** File 6 (psutil — not parallelism, but parallel-adjacent)
**Async I/O:** File 7 (websocket — not computation parallelism)

### Dependencies That Exist Only for Parallel / Concurrency

| Dependency | Location | Purpose | Parallel-only? |
|-----------|----------|---------|---------------|
| `psutil>=5.9.0` | `pyproject.toml` line 45 (under `[streaming]` extra) | System metrics monitoring (`SystemMetricsConnector`) | Not parallel code itself, but only used by streaming feature. Could be removed if streaming is removed. |
| `concurrent.futures` | stdlib | Process/thread pool execution | Yes — used solely for parallel typology computation. |
| `threading` | stdlib | Thread locks for job manager | Yes — used solely for thread-safe job state access. |
| `asyncio` | stdlib | Async WebSocket server | No — async I/O, not parallel computation. |

No third-party packages exist solely for parallelism. `concurrent.futures`, `threading`, and `asyncio` are all Python stdlib.

### CLI Flags Related to Workers / Parallel

| Flag | File | Description |
|------|------|-------------|
| `--workers` | `prime/__main__.py` line 49-50 | `Number of parallel workers for typology (default: PRIME_WORKERS env or 4)` |

### Environment Variables Related to Workers / Parallel

| Variable | File(s) | Description | Default |
|----------|---------|-------------|---------|
| `PRIME_WORKERS` | `prime/ingest/typology_raw.py` line 98, `CLAUDE.md` line 529 | Controls parallel signal processing in typology | Code: `4` (via `int(os.environ.get("PRIME_WORKERS", "0")) or 4`). CLAUDE.md says `1` — **discrepancy**. |
| `MANIFOLD_WORKERS` | `CLAUDE.md` line 532 only | Parallel cohorts in Manifold (not in Prime code) | `0` (auto) |

### Discrepancy Note

`CLAUDE.md` documents `PRIME_WORKERS` default as `1` (single-threaded), but the actual code in `prime/ingest/typology_raw.py:98` defaults to `4` when the env var is unset. The expression `int(os.environ.get("PRIME_WORKERS", "0")) or 4` evaluates to `4` when `PRIME_WORKERS` is not set (because `int("0")` is falsy, so `or 4` kicks in).

### Inaccurate Comment Note

`prime/sql/typology/runner.py` line 355 says `_compute_single_signal` is "Called in parallel via ProcessPoolExecutor" but the actual calling code at line 307 uses `ThreadPoolExecutor`, not `ProcessPoolExecutor`.

---

## Search Patterns Used

The following 19 patterns were searched across all `*.py`, `*.sql`, `*.md`, `*.yaml`, `*.toml`, `*.cfg`, `*.ini`, `*.env` files using ripgrep:

```
1.  ProcessPoolExecutor
2.  ThreadPoolExecutor
3.  multiprocessing
4.  concurrent\.futures
5.  as_completed
6.  _parallel_map
7.  parallel_map
8.  max_workers
9.  n_workers
10. num_workers
11. MANIFOLD_WORKERS
12. parallel/
13. \bfork\b
14. mp_context
15. pool\.submit
16. pool\.map
17. pool\.shutdown
18. psutil
19. cpu_count
```

Additional patterns searched during discovery:

```
20. PRIME_WORKERS
21. executor\.submit
22. \bimport threading\b|\bfrom threading\b
23. \bfrom queue\b|\bimport queue\b
24. \basyncio\b
25. SET threads
26. [Rr]ayon
```
