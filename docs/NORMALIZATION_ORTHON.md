# Normalization in PRISM/Framework

**Version:** 2.5
**Last Updated:** 2026-02-05

> **Primary documentation is in PRISM**: `/Users/jasonrudder/prism/docs/NORMALIZATION.md`
>
> This document summarizes Framework's perspective on normalization choices.

## Framework's Role

```
PRISM computes normalized values.
Framework interprets what they mean and selects the method.
```

Framework's manifest generator can specify the normalization method for PRISM based on signal typology:

| Typology | Recommended Norm | Reason |
|----------|-----------------|--------|
| TRENDING | robust or mad | Trend can look like outliers to z-score |
| IMPULSIVE | mad | Spikes are the signal, not outliers |
| PERIODIC | zscore | Clean, well-defined distribution |
| RANDOM | zscore | Gaussian assumption often valid |
| STATIONARY | zscore | Stable distribution |
| CHAOTIC | mad | Heavy tails common |

## Quick Reference

| Method | Formula | Robustness | Use When |
|--------|---------|------------|----------|
| **zscore** | (x-μ)/σ | Low | Clean Gaussian data |
| **robust** | (x-median)/IQR | Medium | Some outliers |
| **mad** | (x-median)/MAD | High | Industrial, unknown dist |
| **none** | x | N/A | Preserve variance dynamics |

## Anomaly Thresholds

### Z-Score (Sensitive)
```sql
-- prism/engines/sql/zscore.sql
|z| > 3  →  anomaly (0.3% false positive for Gaussian)
```

**Risk:** Outliers inflate std, causing masking.

### MAD Score (Robust)
```sql
-- prism/engines/sql/mad_anomaly.sql
|mad_score| > 3.5  →  anomaly
```

**Severity levels:**
- `CRITICAL`: |m| > 5.0
- `SEVERE`: |m| > 3.5
- `MODERATE`: |m| > 2.5
- `MILD`: |m| > 2.0
- `NORMAL`: |m| ≤ 2.0

## Implementation

Framework SQL classification views (in `framework/sql/classification.sql`) should use MAD-based thresholds when robustness is needed:

```sql
-- Example: Robust anomaly classification
WITH mad_stats AS (
    SELECT
        signal_id,
        MEDIAN(value) AS median_val,
        1.4826 * MEDIAN(ABS(value - median_val)) AS mad_val
    FROM observations
    GROUP BY signal_id
)
SELECT
    *,
    CASE
        WHEN ABS((value - median_val) / NULLIF(mad_val, 0)) > 3.5 THEN 'ANOMALY'
        ELSE 'NORMAL'
    END AS status
FROM observations o
JOIN mad_stats m USING (signal_id)
```

## Files

| Location | Purpose |
|----------|---------|
| `prism/engines/normalization.py` | Core normalization engine |
| `prism/engines/sql/zscore.sql` | Z-score anomaly detection |
| `prism/engines/sql/mad_anomaly.sql` | MAD anomaly detection |
| `prism/docs/NORMALIZATION.md` | Full documentation |
| `framework/sql/classification.sql` | Framework classification (uses PRISM outputs) |

## See Also

- Full documentation: `/Users/jasonrudder/prism/docs/NORMALIZATION.md`
- Typology classification: `framework/typology/`
- Manifest generator: `framework/manifest/generator.py`
