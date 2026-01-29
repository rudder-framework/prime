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

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
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
from orthon.inspection import inspect_file, detect_capabilities, validate_results
from orthon.utils.index_detection import IndexDetector, detect_index, get_index_detection_prompt


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
                "n_rows": int(profile.n_rows) if profile.n_rows is not None else None,
                "n_entities": int(profile.n_entities) if profile.n_entities is not None else None,
                "n_signals": int(profile.n_signals) if profile.n_signals is not None else None,
                "n_timestamps": int(profile.n_timestamps) if profile.n_timestamps is not None else None,
                "min_lifecycle": int(profile.min_lifecycle) if profile.min_lifecycle is not None else None,
                "max_lifecycle": int(profile.max_lifecycle) if profile.max_lifecycle is not None else None,
                "mean_lifecycle": float(profile.mean_lifecycle) if profile.mean_lifecycle is not None else None,
                "median_lifecycle": float(profile.median_lifecycle) if profile.median_lifecycle is not None else None,
                "signal_names": list(profile.signal_names) if profile.signal_names is not None else [],
                "has_nulls": bool(profile.has_nulls) if profile.has_nulls is not None else False,
                "null_pct": float(profile.null_pct) if profile.null_pct is not None else 0.0,
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


@app.post("/api/detect-index")
async def detect_index_column(file: UploadFile = File(...)):
    """
    Detect index column type and sampling interval.

    TIME indices (auto-detectable):
        - ISO 8601: "2024-01-15T14:30:00Z"
        - Unix epoch: 1705329000
        - Date strings: "2024-01-15", "01/15/2024"
        - Excel serial: 45306.604166

    OTHER indices (need user input):
        - Space: requires unit (m, ft, km)
        - Frequency: requires unit (Hz, kHz)
        - Scale: requires unit
        - Cycle: requires duration per cycle

    Returns:
        - column: detected index column name
        - index_type: timestamp, unix_seconds, unix_ms, cycle, spatial, frequency, unknown
        - dimension: time, space, frequency, scale, unknown
        - confidence: high, medium, low
        - needs_user_input: bool
        - sampling_interval_seconds: (for time indices)
        - sampling_unit: seconds, minutes, hours, days
        - sampling_value: numeric value in sampling_unit
        - regularity: regular, mostly_regular, irregular
        - user_prompt: question to ask user (if needs_user_input)
    """
    # Save uploaded file
    suffix = Path(file.filename).suffix if file.filename else '.csv'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        # Read file
        if suffix.lower() == '.parquet':
            df = pl.read_parquet(tmp_path)
        elif suffix.lower() == '.csv':
            df = pl.read_csv(tmp_path)
        elif suffix.lower() in ('.xls', '.xlsx'):
            df = pl.read_excel(tmp_path)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

        # Detect index
        result = detect_index(df)

        # Get prompt for user if needed
        user_prompt = get_index_detection_prompt(result) if result.needs_user_input else None

        response = result.to_dict()
        if user_prompt:
            response['user_prompt'] = user_prompt

        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        tmp_path.unlink(missing_ok=True)


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


# =============================================================================
# CONCIERGE API (LLM-Assisted Workflow)
# =============================================================================

from orthon.concierge import (
    DataConcierge,
    schema_from_dataframe,
    concierge_available,
    get_concierge,
)


@app.get("/api/concierge/status")
async def concierge_status():
    """Check if LLM concierge is available."""
    available = concierge_available()
    return {
        "available": available,
        "message": "Concierge ready" if available else "Set ANTHROPIC_API_KEY to enable AI assistance"
    }


@app.post("/api/concierge/validate")
async def concierge_validate(file: UploadFile = File(...)):
    """
    Validate uploaded data using LLM.

    Returns:
    - Markdown report with findings
    - Suggested configuration JSON
    - List of issues with fixes
    """
    if not concierge_available():
        raise HTTPException(
            status_code=503,
            detail="Concierge not available. Set ANTHROPIC_API_KEY environment variable."
        )

    # Save and read file
    suffix = Path(file.filename).suffix if file.filename else '.csv'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        # Read data
        if suffix.lower() == '.parquet':
            df = pl.read_parquet(tmp_path)
        elif suffix.lower() == '.csv':
            df = pl.read_csv(tmp_path)
        elif suffix.lower() in ('.xls', '.xlsx'):
            df = pl.read_excel(tmp_path)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

        # Create schema info
        schema = schema_from_dataframe(df, filename=file.filename or "uploaded_file")

        # Get concierge validation
        concierge = get_concierge()
        result = concierge.validate_schema(schema)

        return {
            "report": result.report,
            "config": result.config,
            "issues": result.issues,
            "confidence": result.confidence,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/api/concierge/explain-error")
async def concierge_explain_error(
    error_message: str = Form(...),
    error_context: str = Form(""),
    column_name: str = Form(""),
    sample_values: str = Form("")  # JSON array
):
    """
    Explain an error in plain language.

    Returns:
    - summary: What went wrong
    - cause: Likely cause
    - fix_steps: How to fix
    - prevention: How to avoid in future
    """
    if not concierge_available():
        raise HTTPException(
            status_code=503,
            detail="Concierge not available. Set ANTHROPIC_API_KEY."
        )

    try:
        samples = json.loads(sample_values) if sample_values else []
    except:
        samples = []

    concierge = get_concierge()
    explanation = concierge.explain_error(
        error_message=error_message,
        error_context=error_context,
        column_name=column_name,
        sample_values=samples
    )

    return {
        "summary": explanation.summary,
        "cause": explanation.cause,
        "fix_steps": explanation.fix_steps,
        "prevention": explanation.prevention,
    }


@app.post("/api/concierge/interpret")
async def concierge_interpret(
    system_summary: str = Form(...),  # JSON
    signals: str = Form("[]"),  # JSON array
    causal: str = Form("[]"),  # JSON array
    alerts: str = Form("[]"),  # JSON array
):
    """
    Interpret analysis results in plain language.

    Returns:
    - interpretation: Markdown-formatted explanation
    """
    if not concierge_available():
        raise HTTPException(
            status_code=503,
            detail="Concierge not available. Set ANTHROPIC_API_KEY."
        )

    try:
        summary = json.loads(system_summary)
        signal_list = json.loads(signals) if signals else []
        causal_list = json.loads(causal) if causal else []
        alert_list = json.loads(alerts) if alerts else []
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    concierge = get_concierge()
    interpretation = concierge.interpret_results(
        system_summary=summary,
        signal_summaries=signal_list,
        causal_info=causal_list,
        alerts=alert_list
    )

    return {"interpretation": interpretation}


@app.post("/api/concierge/ask")
async def concierge_ask(
    question: str = Form(...),
    signals: str = Form("[]"),
    regimes: str = Form("[]"),
    causal: str = Form("[]"),
    alerts: str = Form("[]"),
    history: str = Form("[]"),  # Previous conversation turns
):
    """
    Answer a question about the analysis.

    Returns:
    - answer: Response text
    """
    if not concierge_available():
        raise HTTPException(
            status_code=503,
            detail="Concierge not available. Set ANTHROPIC_API_KEY."
        )

    try:
        signal_list = json.loads(signals) if signals else []
        regime_list = json.loads(regimes) if regimes else []
        causal_list = json.loads(causal) if causal else []
        alert_list = json.loads(alerts) if alerts else []
        conv_history = json.loads(history) if history else []
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    concierge = get_concierge()
    answer = concierge.answer_question(
        question=question,
        signal_summaries=signal_list,
        regime_info=regime_list,
        causal_info=causal_list,
        alerts=alert_list,
        conversation_history=conv_history
    )

    return {"answer": answer}


# =============================================================================
# SQL WORKFLOW ENDPOINTS (ORTHON SQL Engine)
# =============================================================================
# ORTHON SQL handles:
#   1. observations.parquet - ONLY file ORTHON creates
#   2. Unit-based classification
#   3. PRISM work orders
#   4. Query PRISM results for visualization

import duckdb


def _get_sql_path(filename: str) -> Path:
    """Get path to SQL file."""
    return Path(__file__).parent / "sql" / filename


def _run_sql_file(conn: duckdb.DuckDBPyConnection, filename: str, params: dict = None):
    """Read and execute a SQL file with optional parameter substitution."""
    sql_path = _get_sql_path(filename)
    if not sql_path.exists():
        raise FileNotFoundError(f"SQL file not found: {sql_path}")

    sql_content = sql_path.read_text()

    # Substitute parameters
    if params:
        for key, value in params.items():
            sql_content = sql_content.replace(f"{{{key}}}", str(value))

    # Execute (split by semicolons for multi-statement)
    for statement in sql_content.split(';'):
        statement = statement.strip()
        if statement and not statement.startswith('--') and not statement.startswith('.read'):
            try:
                conn.execute(statement)
            except Exception:
                pass  # Skip invalid statements during multi-statement execution


@app.post("/api/sql/observations")
async def sql_create_observations(
    file: UploadFile = File(...),
    output_dir: str = Form("output"),
):
    """
    Create observations.parquet from uploaded file.

    This is the ONLY parquet file ORTHON creates.
    """
    # Save uploaded file
    suffix = Path(file.filename).suffix if file.filename else '.csv'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        input_path = Path(tmp.name)

    try:
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Run SQL
        conn = duckdb.connect()
        _run_sql_file(conn, "00_observations.sql", {
            "input_path": str(input_path),
            "output_path": str(output_path),
        })

        # Get summary
        summary = conn.execute("SELECT * FROM v_observations_summary").fetchdf()

        return {
            "status": "complete",
            "output_file": str(output_path / "observations.parquet"),
            "summary": summary.to_dict(orient="records")[0] if len(summary) > 0 else {},
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        input_path.unlink(missing_ok=True)


@app.post("/api/sql/classify")
async def sql_classify_signals(
    observations_path: str = Form(...),
):
    """
    Classify signals by UNIT only (no compute).

    Returns signal classifications based on unit-to-class mapping.
    """
    try:
        conn = duckdb.connect()

        # Load observations
        conn.execute(f"CREATE TABLE observations AS SELECT * FROM read_parquet('{observations_path}')")

        # Run classification
        _run_sql_file(conn, "01_classification_units.sql", {})

        # Get results
        summary = conn.execute("SELECT * FROM v_signal_class_summary").fetchdf()
        classes = conn.execute("SELECT * FROM v_signal_class_unit").fetchdf()

        return {
            "status": "complete",
            "summary": summary.to_dict(orient="records"),
            "classifications": classes.to_dict(orient="records"),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/sql/work-orders")
async def sql_get_work_orders(
    observations_path: str = Form(...),
):
    """
    Generate PRISM work orders based on signal classification.

    Returns what PRISM should compute for each signal.
    """
    try:
        conn = duckdb.connect()

        # Load observations
        conn.execute(f"CREATE TABLE observations AS SELECT * FROM read_parquet('{observations_path}')")

        # Run classification + work orders
        _run_sql_file(conn, "01_classification_units.sql", {})
        _run_sql_file(conn, "02_work_orders.sql", {})

        # Get results
        summary = conn.execute("SELECT * FROM v_work_order_summary").fetchdf()
        orders = conn.execute("SELECT * FROM v_prism_work_orders").fetchdf()

        return {
            "status": "complete",
            "summary": summary.to_dict(orient="records"),
            "work_orders": orders.to_dict(orient="records"),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/sql/load-prism-results")
async def sql_load_prism_results(
    prism_output: str = Form(...),  # Directory containing PRISM parquet files
    observations_path: str = Form(None),  # Optional: path to observations.parquet
):
    """
    Load PRISM results for visualization.

    Expects PRISM to have created:
    - signal_typology.parquet
    - behavioral_geometry.parquet
    - dynamical_systems.parquet
    - causal_mechanics.parquet
    """
    prism_path = Path(prism_output)
    if not prism_path.exists():
        raise HTTPException(status_code=404, detail=f"PRISM output not found: {prism_output}")

    # Check for expected files
    expected_files = [
        "signal_typology.parquet",
        "behavioral_geometry.parquet",
        "dynamical_systems.parquet",
        "causal_mechanics.parquet",
    ]

    found = {f: (prism_path / f).exists() for f in expected_files}
    missing = [f for f, exists in found.items() if not exists]

    if missing:
        raise HTTPException(
            status_code=404,
            detail=f"Missing PRISM results: {missing}. PRISM may still be computing."
        )

    try:
        conn = duckdb.connect()

        # Load observations if provided
        if observations_path:
            conn.execute(f"CREATE TABLE observations AS SELECT * FROM read_parquet('{observations_path}')")
            _run_sql_file(conn, "01_classification_units.sql", {})

        # Load PRISM results
        _run_sql_file(conn, "03_load_prism_results.sql", {
            "prism_output": str(prism_path),
        })

        # Get verification
        counts = conn.execute("""
            SELECT
                (SELECT COUNT(*) FROM signal_typology) AS typology_rows,
                (SELECT COUNT(*) FROM behavioral_geometry) AS geometry_rows,
                (SELECT COUNT(*) FROM dynamical_systems) AS dynamics_rows,
                (SELECT COUNT(*) FROM causal_mechanics) AS causality_rows
        """).fetchdf()

        return {
            "status": "loaded",
            "prism_output": str(prism_path),
            "files_found": found,
            "row_counts": counts.to_dict(orient="records")[0] if len(counts) > 0 else {},
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sql/dashboard")
async def sql_get_dashboard(
    prism_output: str,
    observations_path: str = None,
):
    """
    Get dashboard data for visualization.

    Returns system health, alerts, and signal summaries.
    """
    try:
        conn = duckdb.connect()

        # Load data
        if observations_path:
            conn.execute(f"CREATE TABLE observations AS SELECT * FROM read_parquet('{observations_path}')")
            _run_sql_file(conn, "01_classification_units.sql", {})

        _run_sql_file(conn, "03_load_prism_results.sql", {"prism_output": prism_output})
        _run_sql_file(conn, "04_visualization.sql", {})
        _run_sql_file(conn, "05_summaries.sql", {})

        # Get dashboard data
        health = conn.execute("SELECT * FROM v_dashboard_system_health").fetchdf()
        alerts = conn.execute("SELECT * FROM v_dashboard_alerts LIMIT 20").fetchdf()
        layer_summary = conn.execute("SELECT * FROM v_summary_all_layers").fetchdf()

        return {
            "system_health": health.to_dict(orient="records")[0] if len(health) > 0 else {},
            "alerts": alerts.to_dict(orient="records"),
            "layer_summary": layer_summary.to_dict(orient="records"),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sql/signals")
async def sql_get_signals(
    prism_output: str,
    observations_path: str = None,
):
    """
    Get signal analysis cards for the UI.
    """
    try:
        conn = duckdb.connect()

        if observations_path:
            conn.execute(f"CREATE TABLE observations AS SELECT * FROM read_parquet('{observations_path}')")
            _run_sql_file(conn, "01_classification_units.sql", {})

        _run_sql_file(conn, "03_load_prism_results.sql", {"prism_output": prism_output})
        _run_sql_file(conn, "04_visualization.sql", {})

        signals = conn.execute("SELECT * FROM v_dashboard_signal_cards").fetchdf()

        return {
            "signals": signals.to_dict(orient="records"),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sql/correlations")
async def sql_get_correlations(
    prism_output: str,
    min_correlation: float = 0.3,
):
    """
    Get correlation data for heatmap visualization.
    """
    try:
        conn = duckdb.connect()

        _run_sql_file(conn, "03_load_prism_results.sql", {"prism_output": prism_output})
        _run_sql_file(conn, "04_visualization.sql", {})

        correlations = conn.execute(f"""
            SELECT * FROM v_chart_correlation_matrix
            WHERE ABS(correlation) >= {min_correlation}
        """).fetchdf()

        return {
            "correlations": correlations.to_dict(orient="records"),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sql/causal-graph")
async def sql_get_causal_graph(
    prism_output: str,
):
    """
    Get causal graph data (nodes and edges) for visualization.
    """
    try:
        conn = duckdb.connect()

        _run_sql_file(conn, "03_load_prism_results.sql", {"prism_output": prism_output})
        _run_sql_file(conn, "04_visualization.sql", {})

        nodes = conn.execute("SELECT * FROM v_graph_nodes").fetchdf()
        edges = conn.execute("SELECT * FROM v_graph_edges").fetchdf()

        return {
            "nodes": nodes.to_dict(orient="records"),
            "edges": edges.to_dict(orient="records"),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# MANIFEST-BASED COMPUTE (Orthon as Brain)
# =============================================================================

from orthon.services.compute_pipeline import ComputePipeline, get_compute_pipeline
from orthon.services.job_manager import JobManager, JobStatus, get_job_manager
from orthon.intake.config_generator import generate_manifest
from orthon.intake.transformer import prepare_for_prism
from pydantic import BaseModel
from typing import List


class PRISMCallbackPayload(BaseModel):
    """Payload PRISM sends when job completes."""
    job_id: str
    status: str  # "complete" | "failed"
    outputs: List[str] = []  # List of output filenames
    error: str | None = None  # Error message if failed


@app.post("/api/compute/submit")
async def compute_submit(
    file: UploadFile = File(...),
    window_size: int = Form(100),
    window_stride: int = Form(50),
    constants: Optional[str] = Form(None),
):
    """
    Submit data for PRISM computation using manifest architecture.

    Orthon is the brain:
    1. Transforms data to observations.parquet
    2. Analyzes data (units, signals, sampling)
    3. Builds manifest (what PRISM should compute)
    4. Submits to PRISM
    5. Returns job_id for tracking

    Args:
        file: CSV, Excel, or Parquet file
        window_size: Window size for analysis
        window_stride: Stride between windows
        constants: JSON string of global constants
    """
    # Save uploaded file
    suffix = Path(file.filename).suffix if file.filename else '.csv'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        # Parse constants
        parsed_constants = {}
        if constants:
            try:
                parsed_constants = json.loads(constants)
                parsed_constants = {k: v for k, v in parsed_constants.items() if v is not None and v != ""}
            except json.JSONDecodeError:
                pass

        # Transform to observations.parquet
        output_dir = tempfile.mkdtemp(prefix="orthon_job_")
        obs_path, config_path = prepare_for_prism(tmp_path, output_dir)

        # Submit via pipeline
        pipeline = get_compute_pipeline()
        job = await pipeline.submit_async(
            observations_path=obs_path,
            user_id="api_user",
            window_size=window_size,
            window_stride=window_stride,
            constants=parsed_constants,
        )

        return {
            "job_id": job.job_id,
            "status": job.status.value,
            "message": "Job submitted to PRISM",
            "manifest_summary": {
                "engine_count": len(job.manifest.get("engines", [])),
                "categories": job.analysis.get("categories", []),
                "signals": job.analysis.get("signal_count", 0),
                "entities": job.analysis.get("entity_count", 0),
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        tmp_path.unlink(missing_ok=True)


@app.get("/api/compute/jobs/{job_id}")
async def compute_job_status(job_id: str):
    """
    Get status of a compute job.

    Returns current status and progress info.
    """
    job_manager = get_job_manager()
    job = job_manager.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return {
        "job_id": job.job_id,
        "status": job.status.value,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "completed_at": job.completed_at,
        "error": job.error,
        "outputs": job.outputs,
    }


@app.get("/api/compute/jobs/{job_id}/results")
async def compute_job_results(job_id: str):
    """
    Get results of a completed job.

    Returns paths to result parquets and job metadata.
    """
    pipeline = get_compute_pipeline()

    try:
        results = pipeline.get_results(job_id)
        return results
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/compute/jobs")
async def compute_list_jobs(
    status: Optional[str] = None,
    limit: int = 20,
):
    """
    List compute jobs.

    Args:
        status: Filter by status (pending, queued, running, complete, failed)
        limit: Max jobs to return
    """
    job_manager = get_job_manager()

    status_enum = None
    if status:
        try:
            status_enum = JobStatus(status)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid status: {status}")

    jobs = job_manager.list_jobs(status=status_enum, limit=limit)

    return {
        "jobs": [
            {
                "job_id": j.job_id,
                "status": j.status.value,
                "created_at": j.created_at,
                "completed_at": j.completed_at,
            }
            for j in jobs
        ]
    }


# =============================================================================
# PRISM CALLBACK ENDPOINT
# =============================================================================

@app.post("/api/callbacks/prism/{job_id}/complete")
async def prism_callback(
    job_id: str,
    payload: PRISMCallbackPayload,
    background_tasks: BackgroundTasks,
):
    """
    Callback endpoint for PRISM to notify job completion.

    PRISM calls this when it finishes executing a manifest.
    Orthon then:
    1. Updates job status
    2. Records outputs
    3. Notifies user (if configured)

    This is the "ring the bell" mechanism.
    """
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"PRISM callback received for job {job_id}: {payload.status}")

    # Validate job exists
    job_manager = get_job_manager()
    job = job_manager.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    # Handle callback via pipeline
    pipeline = get_compute_pipeline()

    try:
        updated_job = pipeline.handle_callback(
            job_id=job_id,
            status=payload.status,
            outputs=payload.outputs,
            error=payload.error,
        )

        return {
            "status": "accepted",
            "job_status": updated_job.status.value,
            "message": f"Job {job_id} marked as {updated_job.status.value}",
        }

    except Exception as e:
        logger.exception(f"Error handling callback for job {job_id}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/manifest/generate")
async def generate_manifest_endpoint(
    file: UploadFile = File(...),
    window_size: int = Form(100),
    window_stride: int = Form(50),
    constants: Optional[str] = Form(None),
):
    """
    Generate a manifest from uploaded data (without submitting to PRISM).

    Useful for previewing what engines will run.
    """
    suffix = Path(file.filename).suffix if file.filename else '.csv'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        # Parse constants
        parsed_constants = {}
        if constants:
            try:
                parsed_constants = json.loads(constants)
                parsed_constants = {k: v for k, v in parsed_constants.items() if v is not None and v != ""}
            except json.JSONDecodeError:
                pass

        # Transform to observations.parquet
        output_dir = tempfile.mkdtemp(prefix="orthon_manifest_")
        obs_path, _ = prepare_for_prism(tmp_path, output_dir)

        # Generate manifest (but don't submit)
        manifest = generate_manifest(
            obs_path,
            output_dir=output_dir,
            window_size=window_size,
            window_stride=window_stride,
            constants=parsed_constants,
        )

        return {
            "manifest": manifest.model_dump(),
            "summary": manifest.summary(),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        tmp_path.unlink(missing_ok=True)


# =============================================================================
# SIMPLE MANIFEST API (Grouped Engine Format)
# =============================================================================

from orthon.services.manifest_builder import (
    config_to_manifest,
    build_manifest_from_data,
    build_manifest_from_units,
)


@app.post("/api/manifest/build")
async def build_manifest(data: dict):
    """
    Build a simple PRISM manifest from config.

    Input (config):
    {
        "signals": [
            {"name": "P1", "physical_quantity": "pressure", "unit": "PSI"},
            {"name": "T1", "physical_quantity": "temperature", "unit": "degC"}
        ],
        "windows": {"size": 100, "stride": 50},
        "layers": {"typology": true, "geometry": true, "dynamics": true},
        "engines": {"core": ["hurst", "entropy", "lyapunov"]}
    }

    Output (manifest):
    {
        "engines": {
            "signal": ["hurst", "entropy", ...],
            "pair": ["granger", "transfer_entropy"],
            "symmetric_pair": ["correlation", "mutual_info"],
            "windowed": ["rolling_mean", "rolling_std", ...],
            "sql": ["zscore", "statistics"]
        },
        "params": {
            "harmonics": {"sample_rate": 0.1},
            "rolling_mean": {"window": 100}
        }
    }
    """
    try:
        manifest = config_to_manifest(data)
        return manifest
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/manifest/from-signals")
async def manifest_from_signals(data: dict):
    """
    Build manifest directly from signal definitions.

    Input:
    {
        "signals": [
            {"name": "pressure_1", "physical_quantity": "pressure", "unit": "PSI"},
            {"name": "temp_1", "physical_quantity": "temperature", "unit": "degC"}
        ],
        "window_size": 100,
        "stride": 50
    }
    """
    try:
        signals = data.get("signals", [])
        window_size = data.get("window_size", 100)
        stride = data.get("stride", window_size // 2)
        layers = data.get("layers", None)
        core_engines = data.get("core_engines", None)

        manifest = build_manifest_from_data(
            signals=signals,
            window_size=window_size,
            stride=stride,
            layers=layers,
            core_engines=core_engines,
        )

        return manifest

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/manifest/from-units")
async def manifest_from_units(data: dict):
    """
    Build manifest from detected unit categories.

    Input:
    {
        "unit_categories": ["pressure", "temperature", "vibration"],
        "window_size": 100,
        "stride": 50,
        "include_universal": true,
        "include_causality": true
    }

    This is the simplest way to build a manifest - just provide
    the detected unit categories and get back the appropriate engines.
    """
    try:
        unit_categories = data.get("unit_categories", [])
        window_size = data.get("window_size", 100)
        stride = data.get("stride", window_size // 2)
        include_universal = data.get("include_universal", True)
        include_causality = data.get("include_causality", True)

        manifest = build_manifest_from_units(
            unit_categories=unit_categories,
            window_size=window_size,
            stride=stride,
            include_universal=include_universal,
            include_causality=include_causality,
        )

        return manifest

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


def main():
    """CLI entry point for running the server."""
    import uvicorn
    uvicorn.run("orthon.api:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
