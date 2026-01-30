# CARE Wind Turbine Dataset

## Manual Download Required

This dataset requires manual download from the MDPI Data repository.

### Source
- Paper: https://www.mdpi.com/2306-5729/9/12/138
- Title: "CARE to Compare: A Real-World Benchmark Dataset for AI-based Predictive Maintenance in Wind Turbines"

### Dataset Details
- 36 wind turbines across 3 wind farms
- 89 years cumulative operational data
- 44 anomalous turbine-periods + 51 normal periods
- SCADA data (10-minute aggregations typical)

### Download Instructions

1. Visit the MDPI paper supplementary materials
2. Follow the data access link (may require registration)
3. Download all CSV files
4. Place them in: `/data/benchmarks/care_wind/raw/`

### Expected File Structure

```
raw/
├── turbine_1.csv
├── turbine_2.csv
├── ...
├── turbine_36.csv
└── labels.csv (or similar)
```

### After Download

Run:
```bash
python prepare_care_wind.py
```

This will create:
- `observations.parquet`
- `manifest.yaml`

### Notes

- Data is anonymized (turbine/farm identifiers changed)
- Evaluation metric: CARE score (custom metric from paper)
- Ground truth: Binary anomaly labels for each turbine-period
