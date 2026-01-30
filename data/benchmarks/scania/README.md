# SCANIA Component X Dataset

## Manual Download Required

This dataset requires registration for the IDA 2024 Industrial Challenge.

### Source
- Challenge: https://www.ida2024.org/challenge
- IDA 2024 Conference: 27th International Symposium on Intelligent Data Analysis

### Dataset Details
- 33,000+ heavy-duty SCANIA trucks
- Real operational fleet data
- Anonymized component identifier ("Component X")
- Histograms and counters from truck ECU

### Download Instructions

1. Register at: https://www.ida2024.org/challenge
2. Accept terms and conditions
3. Download the challenge dataset
4. Place files in: `/data/benchmarks/scania/raw/`

### Expected File Structure

```
raw/
├── train.csv (or similar)
├── test.csv
├── features.txt (feature descriptions)
└── README.txt (challenge instructions)
```

### After Download

Run:
```bash
python prepare_scania.py
```

This will create:
- `observations.parquet`
- `manifest.yaml`

### Notes

- Features are anonymized (histogram bins, counters)
- Ground truth: Binary failure label for "Component X"
- Large dataset - may need batch processing
- Challenge metric: custom scoring from IDA 2024
