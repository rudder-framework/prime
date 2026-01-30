# ORTHON Benchmark Datasets

Prepared benchmark datasets for PRISM/ORTHON validation.

**DO NOT RUN PRISM** until data has been verified.

## Status

| Dataset | Download | Observations | Entities | Ready |
|---------|----------|--------------|----------|-------|
| FEMTO Bearing | Complete (1.1GB) | 55M | 17 | Yes |
| IMS Bearing | Complete (1.0GB) | Processing | 12 | Processing |
| MetroPT | Downloading (1.6GB) | - | 1 | Pending |
| CARE Wind | Manual | - | 36 | Manual download required |
| SCANIA | Manual | - | 33K+ | Manual download required |

## Directory Structure

```
benchmarks/
├── femto/
│   ├── raw/                    # Downloaded FEMTO data
│   ├── observations.parquet    # 55M observations
│   ├── manifest.yaml           # PRISM configuration
│   └── prepare_femto.py        # Preparation script
├── ims/
│   ├── raw/                    # Downloaded IMS data
│   ├── observations.parquet    # ~250M observations (estimated)
│   ├── manifest.yaml           # PRISM configuration
│   └── prepare_ims.py          # Preparation script
├── metropt/
│   ├── raw/                    # Downloaded MetroPT data
│   ├── observations.parquet    # ~11M observations
│   ├── manifest.yaml           # PRISM configuration
│   └── prepare_metropt.py      # Preparation script
├── care_wind/
│   ├── raw/                    # Manual download required
│   ├── README.md               # Download instructions
│   └── prepare_care_wind.py    # Preparation script
└── scania/
    ├── raw/                    # Manual download required
    ├── README.md               # Download instructions
    └── prepare_scania.py       # Preparation script
```

## Dataset Details

### 1. FEMTO Bearing (Priority 1)
- **Source**: IEEE PHM 2012 Challenge
- **Domain**: Bearing run-to-failure
- **Entities**: 17 bearings (6 training, 11 test)
- **Observations**: 55,022,080
- **Signals**: acc_x, acc_y (25.6 kHz)
- **Operating Conditions**: 3 (1800/1650/1500 RPM with varying loads)

### 2. IMS Bearing (Priority 2)
- **Source**: NASA/University of Cincinnati
- **Domain**: Bearing run-to-failure
- **Entities**: 12 (3 tests × 4 bearings)
- **Signals**: acc_1, acc_2 (20 kHz)
- **Known Failures**: 4 bearings with labeled failure modes

### 3. MetroPT (Priority 3)
- **Source**: Porto Metro APU / Zenodo
- **Domain**: Train compressor
- **Signals**: Pressure, temperature, current, etc.
- **Challenge**: 2+ hour early warning requirement

### 4. CARE Wind (Priority 4)
- **Source**: MDPI Data
- **Domain**: Wind turbine SCADA
- **Entities**: 36 turbines
- **Manual Download Required**: See care_wind/README.md

### 5. SCANIA (Priority 5)
- **Source**: IDA 2024 Challenge
- **Domain**: Truck component failure
- **Entities**: 33,000+ vehicles
- **Manual Download Required**: See scania/README.md

## Verification Checklist

Before running PRISM:

- [ ] FEMTO: 17 entities, 55M observations, no null signals
- [ ] IMS: 12 entities, known failures identified
- [ ] MetroPT: Sampling rate verified, failure labels located
- [ ] CARE Wind: 36 turbines, anomaly labels present
- [ ] SCANIA: Vehicle IDs present, target column identified

## Running Preparation

```bash
# FEMTO (complete)
cd femto && python3 prepare_femto.py

# IMS (processing)
cd ims && python3 prepare_ims.py

# MetroPT (after download completes)
cd metropt && python3 prepare_metropt.py

# CARE Wind (after manual download)
cd care_wind && python3 prepare_care_wind.py

# SCANIA (after registration and download)
cd scania && python3 prepare_scania.py
```

## Notes

- All datasets prepared for PRISM window-based analysis
- Manifest files contain PRISM configuration
- DO NOT RUN PRISM until verification complete
