# PR: Compute Audit & Optimization — SQL First, Rust Last

**Date:** 2026-02-28 (revised)
**Target:** Prime + Machine full compute stack
**Priority:** Post-publication — design now, execute after papers ship
**Depends on:** Modality testing protocol complete ✅ (36/36 pass, Feb 28)
**Revision notes:** Fixed retired "PRISM" references (→ packages/ engines). Corrected Rust Spearman algorithm (global pre-ranking is wrong — must use within-window ranking). Removed false positives where polars already handles the computation.

---

## Philosophy

The performance stack has a strict hierarchy. Violations of this hierarchy are the #1 source of unnecessary compute time:

```
FASTEST:  DuckDB/SQL  — compiled C++, vectorized, columnar, auto-parallel
FAST:     packages/ engines (numpy/scipy) — C backends, but Python call overhead per invocation
SLOW:     Python loops — interpreted, GIL-bound, ~100× slower than compiled
SLOWEST:  Python loops calling scipy per iteration — worst of both worlds
```

**The canonical rule:** `packages/` engines compute pure math (SVD, fingerprinting, similarity). SQL/DuckDB does EVERYTHING ELSE: windowing, aggregations, joins, rolling stats, normalization, feature assembly.

**Fix order:**
1. Move compute from Python → SQL (biggest wins, easiest changes)
2. Optimize remaining Python (vectorize, cache, eliminate waste)
3. Move irreducible compute from Python → Rust (surgical, targeted)

---

## PART 1: SQL AUDIT — WHERE IS PYTHON DOING SQL'S JOB?

### Audit Method

For every computation in Prime and Machine that is NOT pure linear algebra (SVD, matrix multiply, eigendecomposition, distribution fitting):

1. **Can this be expressed as a SQL query?** If yes → it should be DuckDB.
2. **Is it currently in Python?** If yes → performance bug. Flag it.
3. **Estimate speedup** from moving to DuckDB.
4. **Write the replacement query.**

### Already Resolved (No Action Needed)

These were flagged in the initial audit but CC's code review confirmed they're already handled:

**D1/D2 Derivatives — ALREADY POLARS ✅**

`ml_export/derivatives.py:68-69` and `modality/engine.py:202-204` both use:
```python
pl.col(col).diff(n=1).over(group_cols)       # D1
pl.col(col).diff(n=1).diff(n=1).over(group_cols)  # D2
```
Polars `.diff().over()` compiles to the same execution plan as `LAG(x,1) OVER (PARTITION BY cohort ORDER BY signal_0)`. No migration needed.

**Modality Coupling Ranking — ALREADY VECTORIZED ✅**

`modality/engine.py:356` uses `sliding_window_view` + per-row `argsort(argsort())`. Already vectorized numpy, no Python loop over windows. This cannot be moved to SQL (within-window ranking is not a standard SQL window function — see Spearman correction in Part 4).

---

### Audit Target 1.1: Rolling Statistics (Assembler)

**What to look for:** Rolling mean, std, min, max, median over sliding windows per cohort.

**Python anti-pattern:**
```python
for cohort_id in cohort_ids:
    subset = df[df['cohort'] == cohort_id]
    subset['x_rm20'] = subset['x'].rolling(20).mean()
    subset['x_rs20'] = subset['x'].rolling(20).std()
```

**SQL replacement:**
```sql
SELECT
    cohort, signal_0,
    AVG(x) OVER w AS x_rm20,
    STDDEV_SAMP(x) OVER w AS x_rs20,
    MIN(x) OVER w AS x_rmin20,
    MAX(x) OVER w AS x_rmax20
FROM observations
WINDOW w AS (
    PARTITION BY cohort
    ORDER BY signal_0
    ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
)
```

**Why SQL wins:** One pass over the data. All cohorts in parallel. All statistics computed simultaneously. No Python loop, no per-cohort overhead, no DataFrame slicing.

**Expected speedup:** 5-50× depending on cohort count and window count.

**Audit action:** `grep -rn "\.rolling\(" prime/ machine/` — classify each hit.

### Audit Target 1.2: Regime Z-Score Normalization

**What to look for:** Per-regime z-scoring of features.

**Python anti-pattern:**
```python
for regime in regimes:
    mask = df['regime'] == regime
    for col in feature_cols:
        mean = df.loc[mask, col].mean()
        std = df.loc[mask, col].std()
        df.loc[mask, col] = (df.loc[mask, col] - mean) / std
```

**SQL replacement:**
```sql
SELECT
    cohort, signal_0, regime,
    (x - AVG(x) OVER (PARTITION BY regime)) /
        NULLIF(STDDEV_SAMP(x) OVER (PARTITION BY regime), 0) AS x_zscore
FROM observations
```

**Expected speedup:** 5-20×.

**Audit action:** Find the regime normalization code in `regime_export.py`. Is it polars groupby (fast) or Python loop (slow)?

### Audit Target 1.3: Feature Assembly (Joins)

**What to look for:** Joining multiple feature parquets into one wide matrix.

**Python anti-pattern:**
```python
df = regime_rt
df = df.merge(canary_rt, on=['cohort', 'signal_0'])
df = df.merge(csv_stats, on=['cohort', 'signal_0'])
df = df.merge(typology, on=['cohort', 'signal_0'])
df = df.merge(modality_rt, on=['cohort', 'signal_0'])
```

**SQL replacement:**
```sql
SELECT *
FROM 'ml_normalized_rt.parquet' AS rt
JOIN 'ml_canary_rt.parquet' AS canary USING (cohort, signal_0)
JOIN 'ml_csv_stats.parquet' AS csv USING (cohort, signal_0)
JOIN 'ml_typology.parquet' AS typ USING (cohort, signal_0)
JOIN 'ml_modality_features.parquet' AS mod USING (cohort, signal_0)
```

**Why SQL wins:** DuckDB reads parquet files directly without loading into memory. Optimizes join order automatically. Pushes down column projections. A chain of merges creates intermediate DataFrames at each step.

**Expected speedup:** 2-10× and significant memory reduction.

**Audit action:** Find the assembler/runner code. Count merges. Check if DuckDB already handles this.

### Audit Target 1.4: RUL Target Computation

**What to look for:** Computing RUL = max_cycle - current_cycle, capped at 125.

**SQL replacement:**
```sql
SELECT
    cohort, signal_0,
    LEAST(
        MAX(signal_0) OVER (PARTITION BY cohort) - signal_0,
        125
    ) AS rul
FROM observations
```

**Audit action:** Find RUL computation. Is it per-cohort loop or vectorized?

### Audit Target 1.5: Canary Reindexing (Ps30 Sorting)

**SQL replacement:**
```sql
SELECT *,
    ROW_NUMBER() OVER (
        PARTITION BY cohort
        ORDER BY Ps30 DESC
    ) AS canary_rank
FROM observations
```

**Audit action:** Check if this is already polars or if it's a Python loop.

### Audit Target 1.6: Parquet as Database Tables

**Critical insight:** DuckDB treats parquet files as tables natively. The entire Prime output directory IS a database.

```sql
-- One query: join all feature parquets + compute rolling stats + RUL
CREATE VIEW features AS
SELECT *
FROM 'output_time/ml/ml_normalized_rt.parquet' AS rt
JOIN 'output_time/ml/ml_csv_stats.parquet' AS csv USING (cohort, signal_0)
JOIN 'output_time/ml/ml_modality_features.parquet' AS mod USING (cohort, signal_0);

SELECT
    cohort, signal_0,
    -- Rolling stats (if not already computed)
    AVG(thermal_rt_centroid_dist) OVER w20 AS thermal_rt_cd_rm20,
    -- RUL target
    LEAST(MAX(signal_0) OVER (PARTITION BY cohort) - signal_0, 125) AS rul
FROM features
WINDOW
    w20 AS (PARTITION BY cohort ORDER BY signal_0 ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)
```

**NOTE:** This query handles joins, rolling stats, and RUL. It does NOT handle SVD/geometry — that is irreducible math that stays in `packages/` engines. Don't conflate the two.

---

## PART 2: SQL AUDIT EXECUTION

### Step 1: Inventory every computation

```bash
# Find all compute patterns in Prime
grep -rn "\.rolling\(" prime/ machine/
grep -rn "\.groupby\(" prime/ machine/
grep -rn "np\.diff\(" prime/ machine/
grep -rn "\.merge\(" prime/ machine/
grep -rn "\.join\(" prime/ machine/
grep -rn "for.*cohort" prime/ machine/
grep -rn "for.*engine" prime/ machine/
grep -rn "\.transform\(" prime/ machine/
grep -rn "\.clip\(" prime/ machine/
grep -rn "\.sort_values\(" prime/ machine/
grep -rn "LAG\|LEAD\|OVER\|WINDOW" prime/ machine/  # already SQL?
grep -rn "duckdb\|DuckDB" prime/ machine/             # existing DuckDB usage
```

### Step 2: Classify each finding

**Prime audit completed 2026-02-28. Result: A+ — no critical migrations needed.**

| File:Line | Operation | Current impl | Classification | Priority |
|-----------|-----------|-------------|----------------|----------|
| `ml_export/regime_export.py:313-346` | Rolling mean/std/min/max per cohort | `pl.col(x).rolling_mean(w).over(group)` | **Already optimized** | None |
| `ml_export/derivatives.py:68-69` | D1/D2 finite differences per cohort | `pl.col(x).diff(n=1).over(group)` | **Already optimized** | None |
| `modality/engine.py:202-204` | D1/D2 for modality distances | `pl.col(x).diff(n=1).over(cohort)` | **Already optimized** | None |
| `modality/engine.py:356` | Rolling Spearman ρ (within-window rank) | `sliding_window_view` + `argsort(argsort())` vectorized numpy | **Already optimized** (SQL ineligible — see §1.9 note) | None |
| `ml_export/runner.py` | Feature assembly (multi-step join chain) | `pl.DataFrame.join(..., how="full")` per step | **Already optimized** | None |
| `ml_export/runner.py` | groupby aggregations | `pl.group_by().agg()` | **Already optimized** | None |
| `cohorts/discovery.py` | Per-cohort SVD for baseline discovery | NumPy SVD in Python loop over cohorts | **Irreducible math** — adaptive baseline requires cohort-specific SVD | None |
| `modality/engine.py` | Per-modality SVD centroid geometry | NumPy SVD, per signal-pair loop | **Irreducible math** — modality geometry requires SVD per modality | Rust Phase 2 |
| `ml/entry_points/baseline.py:32-38` | RUL computation (max_cycle − current_cycle) | pandas `groupby().transform('max')` | **SQL-eligible** (minor — standalone script, not pipeline hot path) | Low |

Mark legend:
- **Already optimized** — polars/vectorized numpy, no action needed
- **SQL-eligible** — can be expressed as DuckDB query; migration recommended
- **Irreducible math** — stays in `packages/` engines (SVD, fingerprint, similarity)

### Step 3: Migrate and benchmark

For each SQL-eligible finding:
```python
import time, duckdb

# Before
t0 = time.perf_counter()
# ... existing Python code
python_time = time.perf_counter() - t0

# After
t0 = time.perf_counter()
con = duckdb.connect()
result = con.sql("...").fetchdf()
sql_time = time.perf_counter() - t0

print(f"Python: {python_time:.3f}s → DuckDB: {sql_time:.3f}s ({python_time/sql_time:.1f}× speedup)")
```

### Step 4: Verify numerical equivalence

```python
import numpy as np
assert np.allclose(python_result.values, sql_result.values, atol=1e-6, equal_nan=True)
```

---

## PART 3: PYTHON OPTIMIZATION (After SQL Migration)

After moving everything possible to SQL, profile what remains.

### Quick Wins (No Rust Required)

#### A. Cache SVD decompositions
```python
# BAD — SVD recomputed in a loop
for cycle in cycles:
    U, S, Vt = np.linalg.svd(baseline)  # same input every time!

# GOOD — compute once, reuse
U, S, Vt = np.linalg.svd(baseline)
projections = (all_cycles - centroid) @ Vt.T  # single BLAS call
```

#### B. Batch matrix operations
```python
# BAD — per-cycle projection loop
for i, cycle in enumerate(cycles):
    projections[i] = (cycle - centroid) @ Vt.T

# GOOD — single matrix multiply
centered = all_cycles - centroid
projections = centered @ Vt.T
distances = np.linalg.norm(centered, axis=1)
```

#### C. Float32 instead of float64
```python
observations = observations.astype(np.float32)
```
Halves memory, doubles SIMD throughput. ML features don't need 15 decimal digits.

#### D. Eliminate redundant I/O
```python
# BAD — same file read in multiple steps
def step_3():
    obs = pl.read_parquet("observations.parquet")
def step_4():
    obs = pl.read_parquet("observations.parquet")  # again!

# GOOD — read once, pass reference
obs = pl.read_parquet("observations.parquet")
step_3(obs)
step_4(obs)
```

#### E. Use polars lazy evaluation
```python
df = (df.lazy()
    .with_columns([
        pl.col("x").rolling_mean(20).alias("x_rm"),
        pl.col("y").rolling_mean(20).alias("y_rm"),
    ])
    .collect())
```

### Profile Remaining Hotspots

```python
import cProfile, pstats
profiler = cProfile.Profile()
profiler.enable()
run_pipeline(domain_path)
profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(30)
```

---

## PART 4: RUST MIGRATION (After SQL + Python Optimization)

### What's Left for Rust

After SQL takes windowing/aggregations/joins and Python quick wins optimize the rest:

| Priority | Function | Why not SQL | Why not numpy | Rust benefit |
|----------|----------|-------------|---------------|-------------|
| 1 | Rolling Spearman ρ | Within-window ranking not expressible as SQL window fn | Current vectorized numpy is correct but O(n·w·log w) | O(n·w·log w) compiled + SIMD + Rayon parallelism across pairs |
| 2 | RT geometry (SVD → project → distances) | SVD not in SQL | numpy is OK but Python orchestration overhead per entity | Single compiled function, Rayon across entities |
| 3 | gaussian_fingerprint | Distribution fitting not in SQL | sklearn GMM has Python overhead per fit | Pure math, parallelize across signals |
| 4 | gaussian_similarity | Pairwise custom metric not in SQL | scipy pdist is C but Python marshaling per cohort | SIMD pairwise, Rayon across cohorts |

### CORRECTED: Rolling Spearman Algorithm

**The error in the previous version:** Global pre-ranking + running-sum Pearson does NOT compute Spearman. Spearman requires within-window ranking. Pre-ranking globally gives a different statistic that depends on values outside each window. The running-sum trick only works for Pearson, not Spearman.

**The current Python implementation IS CORRECT:**
```python
# engine.py:356 — ranks each window row independently
# sliding_window_view creates (n_windows, window_size) matrix
# argsort(argsort()) ranks each row independently
rx = np.argsort(np.argsort(x_wins, axis=1), axis=1)
ry = np.argsort(np.argsort(y_wins, axis=1), axis=1)
# Then Pearson on the per-window ranks
```

**The correct Rust algorithm:**
```rust
// src/coupling.rs
use rayon::prelude::*;

/// Rolling Spearman correlation with WITHIN-WINDOW ranking.
/// O(n · w · log(w)) — same complexity as Python, but compiled + SIMD.
pub fn rolling_spearman(x: &[f32], y: &[f32], window: usize) -> Vec<f32> {
    let n = x.len();
    assert_eq!(n, y.len());

    let mut result = vec![f32::NAN; n];
    if n < window { return result; }

    // Pre-allocate scratch arrays — no allocation per window
    let mut indices: Vec<usize> = (0..window).collect();
    let mut rx = vec![0.0f32; window];
    let mut ry = vec![0.0f32; window];

    for start in 0..=(n - window) {
        let end = start + window;
        let x_win = &x[start..end];
        let y_win = &y[start..end];

        // Rank x within this window: argsort then assign ranks
        indices.iter_mut().enumerate().for_each(|(i, v)| *v = i);
        indices.sort_unstable_by(|&a, &b| x_win[a].partial_cmp(&x_win[b]).unwrap());
        for (rank, &idx) in indices.iter().enumerate() {
            rx[idx] = rank as f32;
        }

        // Rank y within this window
        indices.iter_mut().enumerate().for_each(|(i, v)| *v = i);
        indices.sort_unstable_by(|&a, &b| y_win[a].partial_cmp(&y_win[b]).unwrap());
        for (rank, &idx) in indices.iter().enumerate() {
            ry[idx] = rank as f32;
        }

        // Pearson on within-window ranks
        result[end - 1] = pearson_from_ranks(&rx, &ry);
    }

    result
}

fn pearson_from_ranks(rx: &[f32], ry: &[f32]) -> f32 {
    let n = rx.len() as f32;
    let sum_rx: f32 = rx.iter().sum();
    let sum_ry: f32 = ry.iter().sum();
    let sum_rx2: f32 = rx.iter().map(|v| v * v).sum();
    let sum_ry2: f32 = ry.iter().map(|v| v * v).sum();
    let sum_rxry: f32 = rx.iter().zip(ry).map(|(a, b)| a * b).sum();

    let num = n * sum_rxry - sum_rx * sum_ry;
    let den = ((n * sum_rx2 - sum_rx * sum_rx) * (n * sum_ry2 - sum_ry * sum_ry)).sqrt();

    if den == 0.0 { return 0.0; }
    num / den
}

/// Parallel across all entity × modality pairs
pub fn batch_rolling_spearman(
    pairs: &[(Vec<f32>, Vec<f32>)],
    window: usize,
) -> Vec<Vec<f32>> {
    pairs.par_iter()
        .map(|(x, y)| rolling_spearman(x, y, window))
        .collect()
}
```

**Key differences from the incorrect version:**
- Ranking happens WITHIN each window, not globally
- No running-sum shortcut (that trick only works for Pearson on raw values)
- Same O(n·w·log w) complexity as the Python version
- Speedup comes from: compiled code, no Python interpreter overhead, SIMD-friendly sort,
  Rayon parallelism across entity pairs, pre-allocated scratch arrays (no allocation per window)
- Expected speedup: 10-50× (not 100-1000× as previously claimed — same algorithm, just compiled)

**NOTE on SQL:** `RANK() OVER (PARTITION BY cohort ORDER BY ...)` computes cohort-level global
ranks — the same error as the global pre-ranking approach. Within-window ranking cannot be
expressed as a standard SQL window function. Rolling Spearman stays in `packages/` engines / Rust.
It is not a SQL migration candidate.

### Rust Crate Structure

```
orthon-core/
├── Cargo.toml
├── src/
│   ├── lib.rs                    # Public API
│   ├── coupling.rs               # Rolling Spearman — within-window ranking
│   ├── geometry.rs               # RT geometry: SVD, projection, distances
│   ├── fingerprint.rs            # Gaussian fingerprinting
│   ├── similarity.rs             # Pairwise similarity
│   ├── baseline.rs               # find_stable_baseline
│   └── utils/
│       ├── mod.rs
│       ├── stats.rs              # rank, correlation helpers
│       └── linalg.rs             # SVD wrapper
├── benches/
│   ├── coupling_bench.rs
│   ├── geometry_bench.rs
│   └── fingerprint_bench.rs
└── python/
    ├── pyproject.toml            # maturin build config
    └── src/
        └── lib.rs                # PyO3 bindings
```

### Dependencies

```toml
[dependencies]
ndarray = "0.15"
ndarray-linalg = "0.16"
rayon = "1.8"
statrs = "0.16"
numpy = "0.20"
pyo3 = "0.20"

[dev-dependencies]
criterion = "0.5"
approx = "0.5"
```

### API Boundary: Python ↔ Rust

Numpy arrays in, numpy arrays out. No DataFrames cross the boundary.

```python
import orthon_core

# Rolling Spearman (within-window ranked, matching scipy.stats.spearmanr)
rho = orthon_core.rolling_spearman(x, y, window=20)

# RT geometry (SVD + projection + all distance metrics)
features = orthon_core.rt_geometry(obs_matrix, early_pct=0.25)

# Fingerprint
params = orthon_core.gaussian_fingerprint(signal)

# Similarity
sim_matrix = orthon_core.pairwise_similarity(fingerprints)
```

### Migration Sequence

```
Phase 1: Rolling Spearman → Rust
  - Correct within-window ranking algorithm
  - Drop-in replacement for engine.py _rolling_spearman()
  - Must pass: np.allclose(python_result, rust_result, atol=1e-5)
  - Expected: 10-50× speedup (compiled + Rayon, same algorithm)

Phase 2: RT Geometry → Rust
  - SVD, projection, distance pipeline per entity
  - Rayon parallelism across entities
  - Expected: 10-50× speedup

Phase 3: Fingerprint + Similarity → Rust
  - Parallelize across signals/cohorts
  - Expected: 5-20× speedup

Phase 4: Full pipeline → Rust
  - Single call: orthon_core.full_pipeline(obs, config) → all features
  - Eliminates all Python↔Rust boundary crossings
```

---

## PART 5: ARCHITECTURE DECISIONS

### f32 vs f64
**f32 default, f64 optional.** ML features don't need 15 decimal digits. Doubles SIMD throughput,
halves memory.

### Parallelism granularity
**Parallelize at entity level.** 260 engines = 260 work items. Rayon work-stealing handles load
balancing. Nested parallelism (entities × modalities) for modality compute path.

### Separate crate
**`orthon-core` on crates.io.** Clean boundary. Rust community can use without Python. Other
languages can bind. `pip install orthon` pulls binary via maturin.

### Streaming API (future)
```rust
// Batch (now)
fn rt_geometry(observations: &Array2<f32>, early_pct: f32) -> GeometryResult;

// Streaming (future)
struct IncrementalGeometry { baseline, centroid, vt }
impl IncrementalGeometry {
    fn new(initial: &Array2<f32>) -> Self;
    fn update(&mut self, observation: &Array1<f32>) -> CycleMetrics;
}
```

---

## PART 6: BENCHMARK PROTOCOL

For EVERY optimization (SQL, Python, or Rust):

```
1. Run current version on FD002, record:
   - Wall time (mean ± std, 10 runs)
   - Peak memory
   - Output values

2. Run optimized version on same data, record same.

3. Verify: np.allclose(original, optimized, atol=1e-5)

4. Report: speedup factor, memory change, numerical agreement.
```

---

## PART 7: PROJECTED PERFORMANCE

### After SQL migration only:

| Operation | Current | After SQL | Speedup |
|-----------|---------|-----------|---------|
| Rolling stats (if Python) | ~30-60s | ~2-5s | 10-20× |
| Regime normalization (if Python loop) | ~5-15s | ~1-2s | 5-10× |
| Feature assembly (joins) | ~10-20s | ~2-3s | 5-7× |
| RUL computation | ~2-5s | ~0.2s | 10-25× |

### After SQL + Python + Rust:

| Operation | After Python opt | After Rust | Speedup |
|-----------|-----------------|------------|---------|
| RT geometry | ~15-30s | ~1-3s | 10-15× |
| Fingerprinting | ~15-20s | ~2-4s | 5× |
| Rolling Spearman | ~5-10s* | ~0.5-1s | 10-20× |

*Current Python implementation is already vectorized (sliding_window_view), so this is faster
than the 55s estimate in the original PR.

### Cumulative:

| Stage | Pipeline time | Cumulative speedup |
|-------|--------------|-------------------|
| Current Python | ~5 min | 1× |
| + SQL migration | ~2 min | 2.5× |
| + Python optimization | ~45-60s | 5-7× |
| + Rust (full) | ~5-15s | 20-60× |

---

## SEQUENCING

```
DONE:          Modality implementation ✅
DONE:          Modality testing protocol (36/36) ✅
IN PROGRESS:   Config F validation (Machine CC)
NEXT:          240-seed validation
THEN:          ──────────── AUDIT BEGINS ────────────
AUDIT STEP 1:  grep codebase for SQL-eligible patterns
AUDIT STEP 2:  Classify (SQL-eligible / already optimized / irreducible math)
AUDIT STEP 3:  Migrate to DuckDB, benchmark each
AUDIT STEP 4:  Python quick wins — cache SVD, batch matrix ops, float32
AUDIT STEP 5:  Profile remaining hotspots
AUDIT STEP 6:  Decision: is Rust needed now or later?
RUST PHASE 1:  Rolling Spearman (within-window ranking, Rayon)
RUST PHASE 2:  RT geometry (SVD pipeline, Rayon)
RUST PHASE 3:  Fingerprint + similarity
RUST PHASE 4:  Full pipeline
FUTURE:        Streaming API, edge deployment, pip install orthon
```

---

## THE CANONICAL RULE (Reinforced)

```
1. Can it be a SQL query?            → DuckDB. No exceptions.
2. Is it pure linear algebra?        → numpy BLAS call or Rust packages/ engine.
3. Is it a statistical function?     → scipy/numpy vectorized or Rust packages/ engine.
4. Is it orchestration/config/IO?    → Python.
5. Is it a loop over data in Python? → BUG. Refactor to 1, 2, or 3.
```

Every new function in Prime or Machine must be reviewed against this hierarchy before merging.
