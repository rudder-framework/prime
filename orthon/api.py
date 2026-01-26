"""
ORTHON API Server
=================

Serves:
- Static HTML shell UI at /
- REST API for config/data operations

Usage:
    uvicorn orthon.api:app --reload
    # or
    orthon-serve
"""

from pathlib import Path
from typing import Optional
import tempfile
import json

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import polars as pl

from orthon.data_reader import DataReader, DataProfile
from orthon.config.recommender import ConfigRecommender
from orthon.config.domains import (
    DOMAINS as LEGACY_DOMAINS,
    EQUATION_INFO,
    get_required_inputs,
    get_equations_for_domain,
    validate_inputs,
    generate_config,
)
from orthon.shared import DISCIPLINES
from orthon.prism_client import get_prism_client, prism_status
from orthon.inspect import inspect_file, detect_capabilities, validate_results


# Store last results path for serving
_last_results_path: Optional[Path] = None


app = FastAPI(
    title="ORTHON",
    description="Diagnostic interpreter for PRISM outputs",
    version="0.1.0",
)

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files (HTML shell)
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def root():
    """Serve the HTML shell UI."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "ORTHON API", "docs": "/docs"}


@app.get("/wizard")
async def wizard():
    """Serve the setup wizard UI."""
    wizard_path = STATIC_DIR / "wizard.html"
    if wizard_path.exists():
        return FileResponse(wizard_path)
    return {"message": "Wizard not found", "docs": "/docs"}


@app.post("/api/profile")
async def profile_data(file: UploadFile = File(...)):
    """
    Profile uploaded data file.

    Returns data characteristics and config recommendations.
    """
    # Save uploaded file temporarily
    suffix = Path(file.filename).suffix if file.filename else '.csv'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        # Read and profile
        reader = DataReader()
        reader.read(tmp_path)
        profile = reader.profile_data()

        # Get recommendations
        recommender = ConfigRecommender(profile)
        rec = recommender.recommend()

        return {
            "profile": {
                "n_rows": profile.n_rows,
                "n_entities": profile.n_entities,
                "n_signals": profile.n_signals,
                "n_timestamps": profile.n_timestamps,
                "min_lifecycle": profile.min_lifecycle,
                "max_lifecycle": profile.max_lifecycle,
                "mean_lifecycle": profile.mean_lifecycle,
                "median_lifecycle": profile.median_lifecycle,
                "signal_names": profile.signal_names,
                "has_nulls": profile.has_nulls,
                "null_pct": profile.null_pct,
            },
            "recommendation": {
                "window_size": rec.window.window_size,
                "window_stride": rec.window.window_stride,
                "overlap_pct": rec.window.overlap_pct,
                "n_windows_approx": rec.window.n_windows_approx,
                "confidence": rec.window.confidence,
                "rationale": rec.window.rationale,
                "n_clusters": rec.n_clusters,
                "n_regimes": rec.n_regimes,
                "alternatives": {
                    "conservative": rec.window.conservative,
                    "aggressive": rec.window.aggressive,
                }
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/api/validate-config")
async def validate_config(config: dict):
    """Validate a PRISM configuration."""
    required = ['window_size', 'window_stride']
    missing = [k for k in required if k not in config]

    if missing:
        return {
            "valid": False,
            "errors": [f"Missing required key: {k}" for k in missing]
        }

    errors = []

    if not isinstance(config.get('window_size'), int) or config['window_size'] < 1:
        errors.append("window_size must be a positive integer")

    if not isinstance(config.get('window_stride'), int) or config['window_stride'] < 1:
        errors.append("window_stride must be a positive integer")

    if config.get('window_stride', 0) > config.get('window_size', 0):
        errors.append("window_stride cannot be larger than window_size")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "config": config if not errors else None,
    }


@app.get("/api/capabilities")
async def get_capabilities():
    """Return PRISM capability definitions for the shell UI."""
    return {
        "capabilities": {
            "STATISTICS": {"level": 0, "name": "Statistics", "requires": []},
            "DISTRIBUTION": {"level": 0, "name": "Distribution", "requires": []},
            "STATIONARITY": {"level": 0, "name": "Stationarity", "requires": []},
            "ENTROPY": {"level": 0, "name": "Entropy", "requires": []},
            "MEMORY": {"level": 0, "name": "Memory (Hurst)", "requires": []},
            "SPECTRAL": {"level": 0, "name": "Spectral", "requires": []},
            "RECURRENCE": {"level": 0, "name": "Recurrence", "requires": []},
            "CHAOS": {"level": 0, "name": "Chaos (Lyapunov)", "requires": []},
            "VOLATILITY": {"level": 0, "name": "Volatility", "requires": []},
            "EVENTS": {"level": 0, "name": "Event Detection", "requires": []},
            "GEOMETRY": {"level": 0, "name": "Geometry", "requires": []},
            "DYNAMICS": {"level": 0, "name": "Dynamics", "requires": []},

            "DERIVATIVES": {"level": 1, "name": "Derivatives", "requires": ["labeled"]},
            "SPECIFIC_KE": {"level": 1, "name": "Specific KE", "requires": ["velocity"]},
            "SPECIFIC_PE": {"level": 1, "name": "Specific PE", "requires": ["position"]},

            "KINETIC_ENERGY": {"level": 2, "name": "Kinetic Energy", "requires": ["velocity", "mass"]},
            "POTENTIAL_ENERGY": {"level": 2, "name": "Potential Energy", "requires": ["position", "spring_constant"]},
            "MOMENTUM": {"level": 2, "name": "Momentum", "requires": ["velocity", "mass"]},
            "HAMILTONIAN": {"level": 2, "name": "Hamiltonian", "requires": ["position", "velocity", "mass", "spring_constant"]},
            "LAGRANGIAN": {"level": 2, "name": "Lagrangian", "requires": ["position", "velocity", "mass", "spring_constant"]},

            "WORK": {"level": 3, "name": "Work", "requires": ["position", "force"]},
            "GIBBS": {"level": 3, "name": "Gibbs Free Energy", "requires": ["temperature", "pressure", "volume", "Cp"]},
            "TRANSFER_FN": {"level": 3, "name": "Transfer Function", "requires": ["input", "output"]},
            "GRANGER": {"level": 3, "name": "Granger Causality", "requires": ["multi"]},
            "TRANSFER_ENTROPY": {"level": 3, "name": "Transfer Entropy", "requires": ["multi"]},

            "VORTICITY": {"level": 4, "name": "Vorticity", "requires": ["velocity_field"]},
            "STRAIN": {"level": 4, "name": "Strain Tensor", "requires": ["velocity_field"]},
            "Q_CRITERION": {"level": 4, "name": "Q-Criterion", "requires": ["velocity_field"]},
            "TKE": {"level": 4, "name": "Turbulent KE", "requires": ["velocity_field"]},
            "ENERGY_SPECTRUM": {"level": 4, "name": "E(k) Spectrum", "requires": ["velocity_field"]},
            "REYNOLDS": {"level": 4, "name": "Reynolds Number", "requires": ["velocity_field", "kinematic_viscosity"]},
        },
        "levels": ["Raw Series", "Labeled", "Constants", "Related", "Spatial"],
    }


# =============================================================================
# DISCIPLINE ENDPOINTS (new architecture)
# =============================================================================

@app.get("/api/disciplines")
async def get_disciplines():
    """Return all discipline definitions for physics engine routing."""
    return {"disciplines": DISCIPLINES}


@app.get("/api/disciplines/{discipline}")
async def get_discipline_info(discipline: str):
    """Return info for a specific discipline including requirements and engines."""
    if discipline not in DISCIPLINES:
        raise HTTPException(status_code=404, detail=f"Discipline not found: {discipline}")
    return {
        "discipline": discipline,
        "info": DISCIPLINES[discipline],
    }


# =============================================================================
# LEGACY DOMAIN ENDPOINTS (for backwards compatibility)
# =============================================================================

@app.get("/api/domains")
async def get_domains():
    """Return all domain definitions for domain-specific wizards. (Legacy)"""
    return {"domains": LEGACY_DOMAINS}


@app.get("/api/domains/{domain}")
async def get_domain_equations(domain: str):
    """Return equations and requirements for a specific domain. (Legacy)"""
    if domain not in LEGACY_DOMAINS:
        raise HTTPException(status_code=404, detail=f"Domain not found: {domain}")
    return {
        "domain": LEGACY_DOMAINS[domain],
        "equations": get_equations_for_domain(domain),
    }


@app.post("/api/domains/{domain}/inputs")
async def get_domain_inputs(domain: str, equations: list[str]):
    """Return required inputs for selected equations. (Legacy)"""
    if domain not in LEGACY_DOMAINS:
        raise HTTPException(status_code=404, detail=f"Domain not found: {domain}")
    return {
        "required_inputs": get_required_inputs(equations),
    }


@app.post("/api/domains/{domain}/validate")
async def validate_domain_inputs(domain: str, data: dict):
    """Validate inputs for selected equations. (Legacy)"""
    equations = data.get('equations', [])
    inputs = data.get('inputs', {})
    errors = validate_inputs(equations, inputs)
    return {
        "valid": len(errors) == 0,
        "errors": errors,
    }


@app.post("/api/domains/{domain}/generate-config")
async def generate_domain_config(domain: str, data: dict):
    """Generate PRISM config from wizard inputs. (Legacy)"""
    if domain not in LEGACY_DOMAINS:
        raise HTTPException(status_code=404, detail=f"Domain not found: {domain}")

    equations = data.get('equations', [])
    signals = data.get('signals', [])
    inputs = data.get('inputs', {})

    # Validate first
    errors = validate_inputs(equations, inputs)
    if errors:
        raise HTTPException(status_code=400, detail={"validation_errors": errors})

    config = generate_config(domain, equations, signals, inputs)
    return {"config": config}


# =============================================================================
# INSPECTION ENDPOINTS
# =============================================================================

@app.post("/api/inspect")
async def inspect_uploaded_file(file: UploadFile = File(...)):
    """
    Inspect uploaded file and detect structure.

    Returns:
        - Detected entities, signals, constants
        - Units from column names
        - Capabilities (what can be computed)
    """
    # Save uploaded file temporarily
    suffix = Path(file.filename).suffix if file.filename else '.csv'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        # Inspect file
        inspection = inspect_file(str(tmp_path))

        # Detect capabilities
        capabilities = detect_capabilities(inspection)

        return {
            "inspection": inspection.to_dict(),
            "capabilities": capabilities.to_dict(),
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/api/validate-results")
async def validate_prism_results():
    """
    Validate the last PRISM results.

    Checks:
        - Parquet files are valid (not corrupted)
        - Files are not empty
        - Expected columns present
    """
    global _last_results_path

    if _last_results_path is None:
        raise HTTPException(status_code=404, detail="No results to validate. Run compute first.")

    try:
        validation = validate_results(str(_last_results_path))
        return validation.to_dict()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# PRISM ENDPOINTS
# =============================================================================

@app.get("/api/prism/health")
async def prism_health():
    """Check if PRISM is available."""
    return prism_status()


@app.post("/api/prism/compute")
async def prism_compute(
    file: UploadFile = File(...),
    discipline: str = Form("core"),
    window_size: int = Form(50),
    window_stride: int = Form(25),
    constants: Optional[str] = Form(None),
):
    """
    Send data to PRISM for computation.

    1. Save uploaded file
    2. Call PRISM /compute
    3. Return results location

    Args:
        constants: JSON string of global constants (density, viscosity, etc.)
    """
    # Save uploaded file
    suffix = Path(file.filename).suffix if file.filename else '.csv'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        observations_path = tmp.name

    # Create output directory
    output_dir = tempfile.mkdtemp(prefix="orthon_results_")

    try:
        # Build config
        config = {
            "discipline": discipline,
            "window": {
                "size": window_size,
                "stride": window_stride,
            }
        }

        # Add global constants if provided
        if constants:
            try:
                parsed_constants = json.loads(constants)
                # Filter out empty values
                global_constants = {k: v for k, v in parsed_constants.items() if v is not None and v != ""}
                if global_constants:
                    config["global_constants"] = global_constants
            except json.JSONDecodeError:
                pass  # Ignore malformed JSON

        # Call PRISM
        client = get_prism_client()
        result = client.compute(
            config=config,
            observations_path=observations_path,
            output_dir=output_dir,
        )

        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result.get("message", "PRISM error"))

        # List result parquets
        results_path = Path(result.get("results_path", output_dir))
        parquets = list(results_path.glob("*.parquet"))

        # Store results path for serving
        global _last_results_path
        _last_results_path = results_path

        return {
            "status": "complete",
            "results_path": str(results_path),
            "parquets": [p.name for p in parquets],
            "parquet_urls": [f"/api/prism/results/{p.name}" for p in parquets],
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up input file (but keep results for download)
        Path(observations_path).unlink(missing_ok=True)


@app.get("/api/prism/results/{filename}")
async def get_prism_result(filename: str):
    """Serve a PRISM result parquet file."""
    global _last_results_path

    if _last_results_path is None:
        raise HTTPException(status_code=404, detail="No results available. Run compute first.")

    # Security: prevent path traversal - only allow basename
    import os
    safe_filename = os.path.basename(filename)
    if safe_filename != filename or '..' in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    file_path = _last_results_path / safe_filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Result file not found")

    return FileResponse(
        file_path,
        media_type="application/octet-stream",
        filename=filename,
    )


@app.post("/api/prism/load-results")
async def load_results_from_path(data: dict):
    """
    Load PRISM results from an external directory path.

    This allows viewing results that weren't generated through the compute endpoint,
    such as results from the prism-inbox directory.

    Args:
        data: {"path": "/path/to/results/directory"}

    Returns:
        List of available parquet files and their URLs
    """
    global _last_results_path

    path_str = data.get("path")
    if not path_str:
        raise HTTPException(status_code=400, detail="Missing 'path' in request body")

    results_path = Path(path_str).expanduser().resolve()

    # Security: verify path exists and is a directory
    if not results_path.exists():
        raise HTTPException(status_code=404, detail=f"Path does not exist: {path_str}")

    if not results_path.is_dir():
        raise HTTPException(status_code=400, detail=f"Path is not a directory: {path_str}")

    # Find parquet files
    parquets = list(results_path.glob("*.parquet"))
    if not parquets:
        raise HTTPException(status_code=404, detail=f"No parquet files found in: {path_str}")

    # Set as current results path
    _last_results_path = results_path

    return {
        "status": "loaded",
        "results_path": str(results_path),
        "parquets": [p.name for p in parquets],
        "parquet_urls": [f"/api/prism/results/{p.name}" for p in parquets],
    }


@app.get("/api/prism/results")
async def list_results():
    """List currently loaded results."""
    global _last_results_path

    if _last_results_path is None:
        return {"loaded": False, "message": "No results loaded"}

    parquets = list(_last_results_path.glob("*.parquet"))

    return {
        "loaded": True,
        "results_path": str(_last_results_path),
        "parquets": [p.name for p in parquets],
        "parquet_urls": [f"/api/prism/results/{p.name}" for p in parquets],
    }


def main():
    """CLI entry point for running the server."""
    import uvicorn
    uvicorn.run("orthon.api:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
