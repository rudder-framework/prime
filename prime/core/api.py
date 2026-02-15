"""
Prime API Server
================

Serves:
- Static HTML shell UI at /
- REST API for config/data operations

Usage:
    uvicorn prime.api:app --reload
    # or
    prime-serve
"""

from pathlib import Path
from typing import Optional
import tempfile
import json

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, Response
import polars as pl

from prime.core.data_reader import DataReader, DataProfile
from prime.config.recommender import ConfigRecommender
from prime.config.domains import (
    DOMAINS as LEGACY_DOMAINS,
    EQUATION_INFO,
    get_required_inputs,
    get_equations_for_domain,
    validate_inputs,
    generate_config,
)
from prime.shared import DISCIPLINES
from prime.core.prism_client import get_prism_client, prism_status
from prime.inspection import inspect_file, detect_capabilities, validate_results
from prime.utils.index_detection import IndexDetector, detect_index, get_index_detection_prompt
from prime.services.job_manager import get_job_manager, JobStatus
from prime.services.state_analyzer import get_state_analyzer, StateThresholds
from prime.services.physics_interpreter import (
    get_physics_interpreter,
    set_physics_config,
    PhysicsInterpreter,
)
from prime.shared.physics_constants import PhysicsConstants


# Store last results path for serving
_last_results_path: Optional[Path] = None


app = FastAPI(
    title="Prime",
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
    return {"message": "Prime API", "docs": "/docs"}


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


@app.get("/api/prism/queue")
async def get_queue_status():
    """
    Get job queue status.

    Returns:
        {
            "running": job_id or null,
            "queued": [job_ids...],
            "queue_length": int,
            "current_job": job details or null
        }
    """
    manager = get_job_manager()
    status = manager.get_queue_status()

    # Add current job details if running
    if status["running"]:
        current = manager.get_job(status["running"])
        if current:
            status["current_job"] = {
                "job_id": current.job_id,
                "status": current.status.value,
                "created_at": current.created_at,
            }

    return status


@app.get("/api/prism/job/{job_id}")
async def get_job_status(job_id: str):
    """
    Get status of a specific job.

    Returns:
        Job details including queue position if queued
    """
    manager = get_job_manager()
    job = manager.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    result = job.to_dict()
    result["queue_position"] = manager.get_queue_position(job_id)

    return result


@app.delete("/api/prism/job/{job_id}")
async def cancel_job(job_id: str):
    """
    Cancel a queued job.

    Cannot cancel a job that is already running.
    """
    manager = get_job_manager()

    if manager.cancel_job(job_id):
        return {"status": "cancelled", "job_id": job_id}
    else:
        job = manager.get_job(job_id)
        if job and job.status == JobStatus.RUNNING:
            raise HTTPException(status_code=400, detail="Cannot cancel running job")
        raise HTTPException(status_code=404, detail="Job not found in queue")


# =============================================================================
# STATE ANALYSIS ENDPOINTS
# =============================================================================

@app.get("/api/state/current/{entity_id}")
async def get_current_state(entity_id: str, job_id: str = None, path: str = None):
    """
    Get current state for an entity.

    Args:
        entity_id: Entity identifier
        job_id: Job ID to look up state.parquet
        path: Or provide path to state.parquet directly
    """
    try:
        analyzer = get_state_analyzer(job_id=job_id, state_path=path)
        state = analyzer.get_current_state(entity_id)

        if not state:
            raise HTTPException(404, f"No state data for {entity_id}")

        return state
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/state/trajectory/{entity_id}")
async def get_state_trajectory(
    entity_id: str,
    job_id: str = None,
    path: str = None,
    start_I: float = None,
    end_I: float = None
):
    """Get state trajectory over time."""
    try:
        analyzer = get_state_analyzer(job_id=job_id, state_path=path)
        df = analyzer.get_state_trajectory(entity_id, start_I, end_I)

        return {
            "entity_id": entity_id,
            "n_points": df.height,
            "data": df.to_dicts()
        }
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/state/transitions/{entity_id}")
async def get_state_transitions(
    entity_id: str,
    job_id: str = None,
    path: str = None,
    velocity_threshold: float = None
):
    """Find state transitions (degradation/recovery events)."""
    try:
        analyzer = get_state_analyzer(job_id=job_id, state_path=path)
        transitions = analyzer.find_transitions(entity_id, velocity_threshold)

        return {
            "entity_id": entity_id,
            "n_transitions": len(transitions),
            "transitions": transitions
        }
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/state/anomalies")
async def get_state_anomalies(
    job_id: str = None,
    path: str = None,
    entity_id: str = None,
    distance_threshold: float = None
):
    """Find all anomalous states."""
    try:
        analyzer = get_state_analyzer(job_id=job_id, state_path=path)
        df = analyzer.find_anomalies(entity_id, distance_threshold)

        return {
            "n_anomalies": df.height,
            "anomalies": df.to_dicts()
        }
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/state/summary/{entity_id}")
async def get_state_summary(entity_id: str, job_id: str = None, path: str = None):
    """Get state summary for an entity."""
    try:
        analyzer = get_state_analyzer(job_id=job_id, state_path=path)
        summary = analyzer.summarize_entity(entity_id)

        if not summary:
            raise HTTPException(404, f"No state data for {entity_id}")

        return summary
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/state/fleet")
async def get_fleet_summary(job_id: str = None, path: str = None):
    """Get state summary for entire fleet."""
    try:
        analyzer = get_state_analyzer(job_id=job_id, state_path=path)
        return analyzer.summarize_fleet()
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/state/entities")
async def get_state_entities(job_id: str = None, path: str = None):
    """Get list of entities in state data."""
    try:
        analyzer = get_state_analyzer(job_id=job_id, state_path=path)
        entities = analyzer.get_entities()

        return {
            "n_entities": len(entities),
            "entities": entities
        }
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/state/all-current")
async def get_all_current_states(job_id: str = None, path: str = None):
    """Get current state for all entities."""
    try:
        analyzer = get_state_analyzer(job_id=job_id, state_path=path)
        states = analyzer.get_all_current_states()

        # Sort by state_distance (worst first)
        states = sorted(states, key=lambda x: x.get('state_distance', 0), reverse=True)

        return {
            "n_entities": len(states),
            "states": states
        }
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


# =============================================================================
# PHYSICS INTERPRETATION ENDPOINTS
# =============================================================================
# Detecting Symplectic Structure Loss via Information-Geometric Coherence
#
# The physics stack (top-down):
#   L4: Thermodynamics → Is energy conserved?
#   L3: Mechanics      → Where is energy flowing?
#   L2: Coherence      → Is symplectic structure intact?
#   L1: State          → Phase space position (consequence)
#
# The Prime Signal: dissipating + decoupling + diverging

@app.get("/api/physics/analyze/{entity_id}")
async def physics_analyze_entity(entity_id: str, job_id: str = None, path: str = None):
    """
    Full physics analysis for an entity.

    Analyzes the complete physics stack (L4→L1) and detects the Prime signal.

    The Prime Signal indicates symplectic structure loss:
        dissipating + decoupling + diverging = degradation

    Args:
        entity_id: Entity identifier
        job_id: Job ID to look up physics.parquet
        path: Or provide path to physics.parquet directly
    """
    try:
        interpreter = get_physics_interpreter(job_id=job_id, physics_path=path)
        analysis = interpreter.analyze_system(entity_id)

        if 'error' in analysis:
            raise HTTPException(404, analysis['error'])

        return analysis
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/physics/energy/{entity_id}")
async def physics_energy_budget(entity_id: str, job_id: str = None, path: str = None):
    """
    L4 Thermodynamics: Is energy conserved?

    The starting point of physics analysis. If energy is conserved,
    the system maintains its symplectic structure.

    Returns:
        - energy_conserved: bool
        - energy_trend: 'stable' | 'accumulating' | 'dissipating'
        - dissipation_rate_mean: float
        - entropy_trend: 'stable' | 'increasing' | 'decreasing'
    """
    try:
        interpreter = get_physics_interpreter(job_id=job_id, physics_path=path)
        result = interpreter.analyze_energy_budget(entity_id)

        if result is None:
            raise HTTPException(404, f"No data for entity {entity_id}")

        return {"entity_id": entity_id, "L4_thermodynamics": result}
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/physics/flow/{entity_id}")
async def physics_energy_flow(entity_id: str, job_id: str = None, path: str = None):
    """
    L3 Mechanics: Where is energy flowing?

    Only meaningful if energy is NOT conserved. Identifies energy
    flow patterns, sources, and sinks.

    Returns:
        - energy_distribution: 'distributed' | 'uneven' | 'concentrated'
        - sources: list of signals gaining energy
        - sinks: list of signals losing energy
    """
    try:
        interpreter = get_physics_interpreter(job_id=job_id, physics_path=path)

        flow = interpreter.analyze_energy_flow(entity_id)
        sources_sinks = interpreter.identify_energy_sources_sinks(entity_id)

        if flow is None:
            raise HTTPException(404, f"No data for entity {entity_id}")

        return {
            "entity_id": entity_id,
            "L3_mechanics": {
                "flow": flow,
                "sources_sinks": sources_sinks,
            }
        }
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/physics/coherence/{entity_id}")
async def physics_coherence(entity_id: str, job_id: str = None, path: str = None):
    """
    L2 Coherence: Is the symplectic structure intact? (Eigenvalue-Based)

    Now uses eigenvalue-based metrics for structural coherence analysis.
    Eigenvalue coherence captures STRUCTURE, not just average correlation.

    Returns:
        - coupling_state: 'strongly_coupled' | 'weakly_coupled' | 'decoupled'
        - structure_state: 'unified' | 'clustered' | 'fragmented'
        - is_decoupling: bool (coherence dropping)
        - is_fragmenting: bool (modes splitting apart)
        - current_coherence: float (λ₁/Σλ - spectral coherence)
        - current_effective_dim: float (participation ratio - how many modes)
        - current_eigenvalue_entropy: float (spectral disorder 0-1)
        - baseline_coherence: float
        - baseline_effective_dim: float
        - coherence_vs_baseline: float (ratio)
        - coherence_trend: float
        - effective_dim_trend: float
        - n_signals: int
        - n_pairs: int
    """
    try:
        interpreter = get_physics_interpreter(job_id=job_id, physics_path=path)
        result = interpreter.analyze_coherence(entity_id)

        if result is None:
            raise HTTPException(404, f"No data for entity {entity_id}")

        return {"entity_id": entity_id, "L2_coherence": result}
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/physics/coherence/{entity_id}/interpret")
async def physics_coherence_interpret(entity_id: str, job_id: str = None, path: str = None):
    """
    Human-readable interpretation of coherence state.

    Returns a natural language summary of what's happening with
    the system's coupling structure based on eigenvalue analysis.
    """
    try:
        interpreter = get_physics_interpreter(job_id=job_id, physics_path=path)
        interpretation = interpreter.interpret_coherence_change(entity_id)

        return {
            "entity_id": entity_id,
            "interpretation": interpretation
        }
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/physics/state/{entity_id}")
async def physics_state(entity_id: str, job_id: str = None, path: str = None):
    """
    L1 State: Where is the system in phase space?

    State is the CONSEQUENCE of energy dynamics.
    state_distance = Mahalanobis distance from baseline (using ALL metrics)
    state_velocity = generalized hd_slope

    Returns:
        - is_stable: bool
        - trend: 'stable' | 'converging' | 'diverging'
        - current_distance: float (σ from baseline)
        - current_velocity: float (rate of change)
    """
    try:
        interpreter = get_physics_interpreter(job_id=job_id, physics_path=path)
        result = interpreter.analyze_state(entity_id)

        if result is None:
            raise HTTPException(404, f"No data for entity {entity_id}")

        return {"entity_id": entity_id, "L1_state": result}
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/physics/fleet")
async def physics_fleet_analysis(job_id: str = None, path: str = None):
    """
    Analyze entire fleet for physics anomalies.

    Identifies all entities with the Prime signal (symplectic structure loss).

    Returns:
        - n_entities: int
        - severity_counts: {normal, watch, warning, critical}
        - prime_signals: list of entity_ids with the signal
        - pct_healthy: float
        - entities: list sorted by severity
    """
    try:
        interpreter = get_physics_interpreter(job_id=job_id, physics_path=path)
        return interpreter.analyze_fleet()
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/physics/entities")
async def physics_get_entities(job_id: str = None, path: str = None):
    """Get list of entities in physics data."""
    try:
        interpreter = get_physics_interpreter(job_id=job_id, physics_path=path)
        entities = interpreter.get_entities()

        return {
            "n_entities": len(entities),
            "entities": entities
        }
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/physics/configure")
async def physics_configure(data: dict):
    """
    Configure real physics mode for a job.

    When units and constants are provided, PhysicsInterpreter computes
    real energy (Joules) instead of proxy energy (y² + dy²).

    Proxy physics detects the same DYNAMICS, just without MAGNITUDE.
    Real physics requires:
        - Signal units (e.g., {'Motor_current': 'A', 'Pump_speed': 'rpm'})
        - Domain constants (e.g., mass, moment_of_inertia, inductance)

    Input:
    {
        "job_id": "abc123",
        "signal_units": {
            "Motor_current": "A",
            "Pump_speed": "rpm",
            "Temperature": "degC"
        },
        "constants": {
            "mass": 100.0,
            "moment_of_inertia": 0.5,
            "inductance": 0.01,
            "thermal_mass": 5000.0
        }
    }
    """
    job_id = data.get("job_id")
    if not job_id:
        raise HTTPException(400, "job_id required")

    signal_units = data.get("signal_units", {})

    # Build PhysicsConstants from input
    constants_dict = data.get("constants", {})
    constants = PhysicsConstants(
        mass=constants_dict.get("mass"),
        spring_constant=constants_dict.get("spring_constant"),
        damping_coefficient=constants_dict.get("damping_coefficient"),
        specific_heat=constants_dict.get("specific_heat"),
        thermal_mass=constants_dict.get("thermal_mass"),
        volume=constants_dict.get("volume"),
        density=constants_dict.get("density"),
        inductance=constants_dict.get("inductance"),
        capacitance=constants_dict.get("capacitance"),
        resistance=constants_dict.get("resistance"),
        moment_of_inertia=constants_dict.get("moment_of_inertia"),
    )

    set_physics_config(job_id, signal_units, constants)

    return {
        "status": "configured",
        "job_id": job_id,
        "signal_units": signal_units,
        "constants": constants.to_dict(),
        "message": "Real physics mode enabled for this job"
    }


# Unit to category mapping for engine selection
UNIT_TO_CATEGORY = {
    # Pressure
    'bar': 'pressure', 'PSI': 'pressure', 'kPa': 'pressure', 'Pa': 'pressure',
    'psi': 'pressure', 'mbar': 'pressure', 'MPa': 'pressure',
    # Temperature
    'degC': 'temperature', 'degF': 'temperature', 'K': 'temperature',
    '°C': 'temperature', '°F': 'temperature',
    # Current
    'A': 'current', 'mA': 'current', 'amp': 'current',
    # Voltage
    'V': 'voltage', 'mV': 'voltage', 'kV': 'voltage',
    # Vibration
    'g': 'vibration', 'mm/s': 'vibration', 'm/s2': 'vibration',
    # Flow
    'gpm': 'flow', 'lpm': 'flow', 'm3/s': 'flow', 'L/min': 'flow',
    # Rotation
    'rpm': 'rotation', 'Hz': 'rotation', 'rad/s': 'rotation',
    # Force/Torque
    'N': 'force', 'kN': 'force', 'lbf': 'force',
    'Nm': 'torque', 'ft-lb': 'torque',
    # Power
    'kW': 'power', 'W': 'power', 'hp': 'power',
}


@app.post("/api/prism/compute")
async def prism_compute(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    discipline: str = Form(""),
    window_size: int = Form(50),
    window_stride: int = Form(25),
    constants: Optional[str] = Form(None),
    units: Optional[str] = Form(None),
):
    """
    Send data to PRISM for computation.

    If a job is already running, this job is queued.

    1. Check queue - if job running, queue this one
    2. Transform user data to observations.parquet
    3. Build manifest with engines based on units
    4. Send both to PRISM
    5. Return results (or queue position)

    Args:
        file: User's data file (CSV, Parquet, Excel)
        discipline: Physics discipline (empty = core analysis only)
        window_size: Window size for rolling calculations
        window_stride: Stride between windows
        constants: JSON string of global constants
        units: JSON string mapping column names to units
    """
    from prime.ingest.transform import IntakeTransformer
    from prime.services.manifest_builder import build_manifest_from_units

    manager = get_job_manager()

    # Save uploaded file
    suffix = Path(file.filename).suffix if file.filename else '.csv'
    work_dir = tempfile.mkdtemp(prefix="prime_job_")
    input_path = Path(work_dir) / f"input{suffix}"

    content = await file.read()
    with open(input_path, 'wb') as f:
        f.write(content)

    try:
        # Parse units if provided
        unit_map = {}
        if units:
            try:
                unit_map = json.loads(units)
            except json.JSONDecodeError:
                pass

        # Transform to observations.parquet
        transformer = IntakeTransformer(discipline=discipline if discipline else None)
        obs_path, config_path = transformer.transform(input_path, work_dir)

        # Detect unit categories from the assigned units
        unit_categories = set()
        for col, unit in unit_map.items():
            # Skip index/ignored columns
            if unit in ('index', 'unitless', 'state', 'count', ''):
                continue
            # Map unit to category
            category = UNIT_TO_CATEGORY.get(unit)
            if category:
                unit_categories.add(category)

        # Build manifest
        manifest = build_manifest_from_units(
            unit_categories=list(unit_categories),
            window_size=window_size,
            stride=window_stride,
            include_universal=True,
            include_causality=len(unit_categories) > 1,  # Only if multiple signal types
        )

        # Add discipline-specific engines if selected
        if discipline and discipline not in ("", "core"):
            manifest["discipline"] = discipline

        # Add constants to manifest
        if constants:
            try:
                parsed_constants = json.loads(constants)
                global_constants = {k: v for k, v in parsed_constants.items() if v is not None and v != ""}
                if global_constants:
                    manifest["constants"] = global_constants
            except json.JSONDecodeError:
                pass

        # Create job record
        job = manager.create_job(
            manifest=manifest,
            observations_path=str(obs_path),
            output_dir=work_dir,
        )

        # Check queue and enqueue
        queue_result = manager.enqueue_job(job.job_id)

        if queue_result["status"] == "queued":
            # Job is queued, return immediately with queue position
            return {
                "status": "queued",
                "job_id": job.job_id,
                "queue_position": queue_result["position"],
                "message": f"Job queued at position {queue_result['position']}. A job is currently running.",
            }

        # Job is running - execute now
        client = get_prism_client()
        result = client.compute(
            observations_path=str(obs_path),
            manifest=manifest,
        )

        # Mark job complete or failed
        if result.get("status") == "error":
            manager.complete_current_job(JobStatus.FAILED)
            raise HTTPException(status_code=500, detail=result.get("message", "PRISM error"))

        manager.set_outputs(job.job_id, result.get("files", []), result.get("output_dir", ""))
        manager.complete_current_job(JobStatus.COMPLETE)

        # Store job info for serving results
        global _last_results_path
        _last_results_path = result.get("job_id")

        return {
            "status": "complete",
            "job_id": job.job_id,
            "prism_job_id": result.get("job_id"),
            "files": result.get("files", []),
            "file_urls": result.get("file_urls", []),
            "duration_seconds": result.get("duration_seconds"),
        }

    except HTTPException as e:
        # Mark job as failed and process next in queue
        if 'job' in locals():
            manager.complete_current_job(JobStatus.FAILED)
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        # Mark job as failed and process next in queue
        if 'job' in locals():
            manager.complete_current_job(JobStatus.FAILED)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/prism/results/{job_id}/{filename}")
async def get_prism_result_by_job(job_id: str, filename: str):
    """Proxy a PRISM result parquet file by job_id."""
    import os

    # Security: prevent path traversal
    safe_job_id = os.path.basename(job_id)
    safe_filename = os.path.basename(filename)
    if safe_job_id != job_id or safe_filename != filename:
        raise HTTPException(status_code=400, detail="Invalid path")

    # Fetch from PRISM
    try:
        client = get_prism_client()
        prism_url = f"{client.base_url}/results/{safe_job_id}/{safe_filename}"

        import httpx
        async with httpx.AsyncClient() as async_client:
            r = await async_client.get(prism_url)

        if r.status_code == 404:
            raise HTTPException(status_code=404, detail="Result file not found")

        return Response(
            content=r.content,
            media_type="application/octet-stream",
            headers={"Content-Disposition": f"attachment; filename={safe_filename}"}
        )
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="PRISM server not available")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/prism/results/{filename}")
async def get_prism_result(filename: str):
    """Serve a PRISM result parquet file (legacy - uses last job)."""
    global _last_results_path

    if _last_results_path is None:
        raise HTTPException(status_code=404, detail="No results available. Run compute first.")

    # If _last_results_path is a job_id string, proxy from PRISM
    if isinstance(_last_results_path, str):
        return await get_prism_result_by_job(_last_results_path, filename)

    # Legacy: local path
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

from prime.services.concierge import (
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
# PRIME CONCIERGE (Natural Language -> SQL Reports)
# =============================================================================
# Ask questions in plain English, get Prime analysis

from prime.services.concierge import Concierge as PrimeConcierge, ConciergeResponse


@app.post("/api/prime/ask")
async def prime_ask(data: dict):
    """
    Ask Prime a question in natural language.

    This is the main Concierge endpoint - users can ask questions like:
    - "Which entity is healthiest?"
    - "What's wrong with entity 97?"
    - "Compare all entities"
    - "When did failures start?"
    - "Why is entity 105 degrading?"

    Input:
    {
        "question": "Which entity is in the worst condition?",
        "data_dir": "/path/to/parquet/files"  # Optional, uses last loaded results
    }

    Output:
    {
        "question": "Which entity is in the worst condition?",
        "answer": "**Entity 105** is in the worst condition...",
        "sql": "SELECT ... FROM physics ...",
        "data": [{"entity_id": "105", ...}],
        "confidence": "high"
    }
    """
    global _last_results_path

    question = data.get("question")
    if not question:
        raise HTTPException(status_code=400, detail="Missing 'question' in request body")

    # Get data directory
    data_dir = data.get("data_dir")
    if not data_dir and _last_results_path:
        data_dir = str(_last_results_path)
    if not data_dir:
        raise HTTPException(
            status_code=400,
            detail="No data directory specified. Either provide 'data_dir' or load results first."
        )

    try:
        concierge = PrimeConcierge(data_dir)
        response = concierge.ask(question)

        return {
            "question": response.question,
            "answer": response.answer,
            "sql": response.sql,
            "data": response.data,
            "confidence": response.confidence,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/prime/health/{entity_id}")
async def prime_entity_health(entity_id: str, data_dir: str = None):
    """
    Get health status for a specific entity.

    Quick endpoint for entity-specific queries.
    """
    global _last_results_path

    if not data_dir and _last_results_path:
        data_dir = str(_last_results_path)
    if not data_dir:
        raise HTTPException(status_code=400, detail="No data directory available")

    try:
        concierge = PrimeConcierge(data_dir)
        response = concierge.ask(f"health of entity {entity_id}")

        return {
            "entity_id": entity_id,
            "answer": response.answer,
            "data": response.data,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/prime/fleet")
async def prime_fleet_summary(data_dir: str = None):
    """
    Get fleet-wide summary.

    Quick overview of all entities.
    """
    global _last_results_path

    if not data_dir and _last_results_path:
        data_dir = str(_last_results_path)
    if not data_dir:
        raise HTTPException(status_code=400, detail="No data directory available")

    try:
        concierge = PrimeConcierge(data_dir)
        response = concierge.ask("fleet summary")

        return {
            "answer": response.answer,
            "data": response.data,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/prime/anomalies")
async def prime_anomalies(data_dir: str = None):
    """
    Find anomalous entities.

    Identifies entities deviating from fleet norms.
    """
    global _last_results_path

    if not data_dir and _last_results_path:
        data_dir = str(_last_results_path)
    if not data_dir:
        raise HTTPException(status_code=400, detail="No data directory available")

    try:
        concierge = PrimeConcierge(data_dir)
        response = concierge.ask("find anomalies")

        return {
            "answer": response.answer,
            "data": response.data,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/prime/prime-signal")
async def prime_signal_status(data_dir: str = None):
    """
    Check Prime signal status across fleet.

    The Prime signal = dissipating + decoupling + diverging
    """
    global _last_results_path

    if not data_dir and _last_results_path:
        data_dir = str(_last_results_path)
    if not data_dir:
        raise HTTPException(status_code=400, detail="No data directory available")

    try:
        concierge = PrimeConcierge(data_dir)
        response = concierge.ask("prime signal status")

        return {
            "answer": response.answer,
            "data": response.data,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/suggest-units")
async def suggest_units(data: dict):
    """
    Use AI to suggest units for signal names.

    Input: {"signals": ["TP2", "TP3", "Pump_Speed", "Flow_Rate"]}
    Output: {"TP2": "bar", "TP3": "bar", "Pump_Speed": "rpm", "Flow_Rate": "gpm"}
    """
    if not concierge_available():
        raise HTTPException(
            status_code=503,
            detail="Concierge not available. Set ANTHROPIC_API_KEY."
        )

    signals = data.get("signals", [])
    if not signals:
        return {}

    try:
        concierge = get_concierge()

        # Build prompt for unit suggestion
        prompt = f"""Analyze these signal names from industrial sensor data and suggest appropriate units.

Signal names: {', '.join(signals)}

IMPORTANT: Use EXACTLY these unit values (case-sensitive):

Physical units:
- Temperature: degC, degF, K
- Pressure: bar, PSI, kPa, Pa
- Flow: gpm, lpm, m3/s
- Rotation: rpm, Hz
- Electrical: V, A, kW
- Vibration: g, mm/s
- Level/percentage: %
- Length: m

Digital/Binary signals (on/off, switches, states):
- state (for binary on/off signals, switches, digital states)
- count (for pulse counters, impulse counters)

Special:
- index (for index columns like "Unnamed: 0", row numbers, sequence numbers)
- unitless (for dimensionless ratios)

Pattern hints:
- TP, P_, pressure, reservoir = pressure (bar)
- T_, Temp, temperature, Oil_temp = temperature (degC)
- current, Motor_current, I_ = electrical current (A)
- voltage, V_ = voltage (V)
- speed, RPM = rotation (rpm)
- switch, DV_, COMP, Tower, MPG, LPS = digital state (state)
- impulse, pulse, caudal = pulse counter (count)
- level, Oil_level = could be state (binary) or % (analog)
- Unnamed, index, column00, column with sequential numbers = row index (index)

Respond ONLY with valid JSON mapping signal name to unit. Example:
{{"Unnamed: 0": "index", "TP2": "bar", "Oil_temperature": "degC", "Motor_current": "A", "Pressure_switch": "state"}}
"""

        # Use the concierge's client directly
        response = concierge.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}]
        )

        # Parse the response
        response_text = response.content[0].text.strip()

        # Try to extract JSON from the response
        import re
        json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
        if json_match:
            suggestions = json.loads(json_match.group())
            return suggestions
        else:
            # Fallback: return empty
            return {}

    except Exception as e:
        print(f"Error suggesting units: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# SQL WORKFLOW ENDPOINTS (Prime SQL Engine)
# =============================================================================
# Prime SQL handles:
#   1. observations.parquet - ONLY file Prime creates
#   2. Unit-based classification
#   3. PRISM work orders
#   4. Query PRISM results for visualization

import duckdb

from prime.sql.generate_readme import generate_sql_readme
from prime.io.readme_writer import generate_manifold_readmes


def _get_sql_path(filename: str) -> Path:
    """Get path to SQL file."""
    return Path(__file__).parent.parent / "sql" / filename


def _run_sql_file(conn: duckdb.DuckDBPyConnection, filename: str, params: dict = None,
                   output_dir: Path = None):
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
        # Remove .print/.read directives (DuckDB CLI only)
        lines = [line for line in statement.split('\n')
                 if not line.strip().startswith('.')]
        cleaned = '\n'.join(lines).strip()
        if not cleaned:
            continue
        # Skip blocks that are only comments
        has_sql = any(line.strip() and not line.strip().startswith('--')
                      for line in cleaned.split('\n'))
        if not has_sql:
            continue
        try:
            conn.execute(cleaned)
        except Exception:
            pass  # Skip invalid statements during multi-statement execution


@app.post("/api/sql/observations")
async def sql_create_observations(
    file: UploadFile = File(...),
    output_dir: str = Form("output"),
):
    """
    Create observations.parquet from uploaded file.

    This is the ONLY parquet file Prime creates.
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
        }, output_dir=output_path)

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
            _run_sql_file(conn, "01_classification_units.sql", {}, output_dir=prism_path)

        # Load PRISM results
        _run_sql_file(conn, "03_load_prism_results.sql", {
            "prism_output": str(prism_path),
        }, output_dir=prism_path)

        # Get verification
        counts = conn.execute("""
            SELECT
                (SELECT COUNT(*) FROM signal_typology) AS typology_rows,
                (SELECT COUNT(*) FROM behavioral_geometry) AS geometry_rows,
                (SELECT COUNT(*) FROM dynamical_systems) AS dynamics_rows,
                (SELECT COUNT(*) FROM causal_mechanics) AS causality_rows
        """).fetchdf()

        # Generate READMEs for output directories
        try:
            generate_sql_readme(conn, prism_path)
            generate_manifold_readmes(prism_path)
        except Exception:
            pass  # README generation is non-critical

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
        _out = Path(prism_output)
        if observations_path:
            conn.execute(f"CREATE TABLE observations AS SELECT * FROM read_parquet('{observations_path}')")
            _run_sql_file(conn, "01_classification_units.sql", {}, output_dir=_out)

        _run_sql_file(conn, "03_load_prism_results.sql", {"prism_output": prism_output}, output_dir=_out)
        _run_sql_file(conn, "04_visualization.sql", {}, output_dir=_out)
        _run_sql_file(conn, "05_summaries.sql", {}, output_dir=_out)

        # Generate SQL README alongside output
        try:
            generate_sql_readme(conn, _out)
        except Exception:
            pass  # README generation is non-critical

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

        _out = Path(prism_output)
        if observations_path:
            conn.execute(f"CREATE TABLE observations AS SELECT * FROM read_parquet('{observations_path}')")
            _run_sql_file(conn, "01_classification_units.sql", {}, output_dir=_out)

        _run_sql_file(conn, "03_load_prism_results.sql", {"prism_output": prism_output}, output_dir=_out)
        _run_sql_file(conn, "04_visualization.sql", {}, output_dir=_out)

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

        _out = Path(prism_output)
        _run_sql_file(conn, "03_load_prism_results.sql", {"prism_output": prism_output}, output_dir=_out)
        _run_sql_file(conn, "04_visualization.sql", {}, output_dir=_out)

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

        _out = Path(prism_output)
        _run_sql_file(conn, "03_load_prism_results.sql", {"prism_output": prism_output}, output_dir=_out)
        _run_sql_file(conn, "04_visualization.sql", {}, output_dir=_out)

        nodes = conn.execute("SELECT * FROM v_graph_nodes").fetchdf()
        edges = conn.execute("SELECT * FROM v_graph_edges").fetchdf()

        return {
            "nodes": nodes.to_dict(orient="records"),
            "edges": edges.to_dict(orient="records"),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# MANIFEST-BASED COMPUTE (Prime as Brain)
# =============================================================================

from prime.services.compute_pipeline import ComputePipeline, get_compute_pipeline
from prime.services.job_manager import JobManager, JobStatus, get_job_manager
from prime.ingest.config_generator import generate_manifest
from prime.ingest.transform import prepare_for_prism
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

    Prime is the brain:
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
        output_dir = tempfile.mkdtemp(prefix="prime_job_")
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
    Prime then:
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
        output_dir = tempfile.mkdtemp(prefix="prime_manifest_")
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

from prime.services.manifest_builder import (
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
        "engines": {"core": ["hurst", "entropy"]}
    }

    Output (manifest):
    {
        "engines": {
            "signal": ["hurst", "entropy", ...],
            "pair": ["granger", "transfer_entropy"],
            "symmetric_pair": ["correlation", "mutual_info"],
            "windowed": ["rolling_mean", "rolling_std", ...],
            "sql": ["trajectory_deviation", "statistics"]
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


# =============================================================================
# AI-GUIDED TUNING ENDPOINTS
# =============================================================================
# Validate PRISM detection against ground truth labels.
# Learn optimal thresholds and fault signatures.

from prime.services.tuning_service import TuningService, get_tuning_service
from prime.services.fingerprint_service import (
    FingerprintService,
    get_fingerprint_service,
    generate_healthy_fingerprint,
    generate_deviation_fingerprint,
)


@app.post("/api/tuning/analyze")
async def tuning_analyze(data: dict):
    """
    Analyze detection performance against ground truth.

    This is the main tuning endpoint. Run after PRISM analysis on labeled data.

    Input:
    {
        "data_dir": "/path/to/prism/results",
        "labels_path": "/path/to/labels.parquet"  # Optional if in data_dir
    }

    Output:
    {
        "optimal_z_threshold": 2.5,
        "best_metrics_by_fault_type": {"valve": "coherence", "cavitation": "entropy"},
        "detection_rate": 94.0,
        "avg_lead_time": 47.3,
        "n_entities": 100,
        "metric_rankings": [...],
        "threshold_curve": [...],
        "recommendations": "## Tuning Recommendations..."
    }
    """
    global _last_results_path

    data_dir = data.get("data_dir")
    labels_path = data.get("labels_path")

    # Use last loaded results if no data_dir provided
    if not data_dir and _last_results_path:
        data_dir = str(_last_results_path)

    if not data_dir:
        raise HTTPException(
            status_code=400,
            detail="No data directory specified. Either provide 'data_dir' or load results first."
        )

    try:
        tuner = TuningService(data_dir, labels_path)
        result = tuner.tune()

        return result.to_dict()

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/tuning/optimize-thresholds")
async def tuning_optimize_thresholds(data: dict):
    """
    Find optimal detection thresholds based on ground truth.

    Tests multiple z-thresholds and returns performance at each.

    Input:
    {
        "data_dir": "/path/to/prism/results"
    }

    Output:
    {
        "optimal": {"z": 2.5, "criterion": "best_balanced", "detection_rate": 94.0},
        "threshold_curve": [
            {"z_threshold": 1.5, "detection_rate": 98.0, "avg_lead_time": 60.0, ...},
            {"z_threshold": 2.0, "detection_rate": 95.0, "avg_lead_time": 52.0, ...},
            ...
        ]
    }
    """
    global _last_results_path

    data_dir = data.get("data_dir")
    labels_path = data.get("labels_path")

    if not data_dir and _last_results_path:
        data_dir = str(_last_results_path)

    if not data_dir:
        raise HTTPException(status_code=400, detail="No data directory specified.")

    try:
        tuner = TuningService(data_dir, labels_path)

        optimal = tuner.find_optimal_threshold()
        threshold_curve = tuner.get_threshold_curve()

        return {
            "optimal": optimal,
            "threshold_curve": threshold_curve.to_dicts() if threshold_curve.height > 0 else [],
        }

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/tuning/fault-signatures")
async def tuning_fault_signatures(data: dict):
    """
    Learn fault signatures - which metrics detect which fault types best.

    Input:
    {
        "data_dir": "/path/to/prism/results"
    }

    Output:
    {
        "signatures": {"valve": "coherence", "cavitation": "entropy"},
        "signature_matrix": [
            {"fault_type": "valve", "metric_name": "coherence", "detection_rate": 94.0, ...},
            ...
        ]
    }
    """
    global _last_results_path

    data_dir = data.get("data_dir")
    labels_path = data.get("labels_path")

    if not data_dir and _last_results_path:
        data_dir = str(_last_results_path)

    if not data_dir:
        raise HTTPException(status_code=400, detail="No data directory specified.")

    try:
        tuner = TuningService(data_dir, labels_path)

        signatures = tuner.learn_fault_signatures()
        matrix = tuner.get_fault_signature_matrix()

        return {
            "signatures": signatures,
            "signature_matrix": matrix.to_dicts() if matrix.height > 0 else [],
        }

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/tuning/metric-performance")
async def tuning_metric_performance(data: dict):
    """
    Get per-metric detection performance.

    Shows which metrics have best detection rate and lead time.

    Input:
    {
        "data_dir": "/path/to/prism/results"
    }

    Output:
    {
        "metric_rankings": [
            {"metric_name": "coherence", "detection_rate_pct": 94.0, "avg_lead_time": 47.0, ...},
            {"metric_name": "entropy", "detection_rate_pct": 88.0, "avg_lead_time": 52.0, ...},
            ...
        ]
    }
    """
    global _last_results_path

    data_dir = data.get("data_dir")
    labels_path = data.get("labels_path")

    if not data_dir and _last_results_path:
        data_dir = str(_last_results_path)

    if not data_dir:
        raise HTTPException(status_code=400, detail="No data directory specified.")

    try:
        tuner = TuningService(data_dir, labels_path)
        metric_perf = tuner.get_metric_performance()

        return {
            "metric_rankings": metric_perf.to_dicts() if metric_perf.height > 0 else [],
        }

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/tuning/generate-config")
async def tuning_generate_config(data: dict):
    """
    Generate tuned configuration based on validation results.

    Creates optimized thresholds and priority metrics based on tuning analysis.

    Input:
    {
        "data_dir": "/path/to/prism/results"
    }

    Output:
    {
        "tuned_thresholds": {"z_warning": 2.5, "z_critical": 3.5},
        "priority_metrics": ["coherence", "entropy", "lyapunov"],
        "fault_signatures": {"valve": "coherence", "cavitation": "entropy"},
        "tuning_metadata": {...}
    }
    """
    global _last_results_path

    data_dir = data.get("data_dir")
    labels_path = data.get("labels_path")

    if not data_dir and _last_results_path:
        data_dir = str(_last_results_path)

    if not data_dir:
        raise HTTPException(status_code=400, detail="No data directory specified.")

    try:
        tuner = TuningService(data_dir, labels_path)
        config = tuner.generate_tuned_config()

        return config.to_dict()

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/tuning/labels")
async def tuning_get_labels(data_dir: str = None):
    """
    Get summary of available ground truth labels.

    Returns what label columns are available and their value distributions.
    """
    global _last_results_path

    if not data_dir and _last_results_path:
        data_dir = str(_last_results_path)

    if not data_dir:
        raise HTTPException(status_code=400, detail="No data directory specified.")

    try:
        tuner = TuningService(data_dir)
        label_summary = tuner.get_label_summary()

        return {
            "labels": label_summary.to_dicts() if label_summary.height > 0 else [],
        }

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# FINGERPRINT ENDPOINTS
# =============================================================================
# System health fingerprints - learned signatures for healthy, deviation, failure states.
# Same math. Different fingerprints. Every system has its own signature.


@app.get("/api/fingerprints/domains")
async def fingerprints_list_domains():
    """
    List available fingerprint domains.

    Returns domains that have fingerprints stored.
    """
    fps = get_fingerprint_service()
    domains = []

    if fps.fingerprints_dir.exists():
        for domain_dir in fps.fingerprints_dir.iterdir():
            if domain_dir.is_dir():
                count = len(list(domain_dir.glob("*.yaml")))
                if count > 0:
                    domains.append({
                        "domain": domain_dir.name,
                        "fingerprint_count": count,
                    })

    return {"domains": domains}


@app.get("/api/fingerprints/{domain}")
async def fingerprints_get_domain(domain: str):
    """
    Get all fingerprints for a domain.

    Returns healthy, deviation, and failure fingerprints.
    """
    fps = get_fingerprint_service()
    count = fps.load_fingerprints(domain)

    if count == 0:
        raise HTTPException(status_code=404, detail=f"No fingerprints found for domain: {domain}")

    healthy = fps.healthy.get(domain)
    deviations = fps.deviations.get(domain, [])
    failures = fps.failures.get(domain, [])

    return {
        "domain": domain,
        "healthy": healthy.to_dict() if healthy else None,
        "deviations": [d.to_dict() for d in deviations],
        "failures": [f.to_dict() for f in failures],
    }


@app.post("/api/fingerprints/classify")
async def fingerprints_classify(data: dict):
    """
    Classify current system state against fingerprints.

    Input:
    {
        "domain": "pump",
        "metrics": {
            "coherence": 0.18,
            "lyapunov": -0.03,
            "entropy": 1.45,
            "hurst": 0.58
        }
    }

    Output:
    {
        "state": "deviation",
        "fault_type": "valve_obstruction",
        "confidence": 87.5,
        "lead_time": 47,
        "matching_indicators": ["coherence", "hurst"],
        "description": "Pattern matches valve obstruction..."
    }
    """
    domain = data.get("domain")
    metrics = data.get("metrics", {})

    if not domain:
        raise HTTPException(status_code=400, detail="Missing 'domain' in request")
    if not metrics:
        raise HTTPException(status_code=400, detail="Missing 'metrics' in request")

    fps = get_fingerprint_service()
    fps.load_fingerprints(domain)

    match = fps.classify_state(metrics, domain)

    return {
        "state": match.fingerprint_type,
        "fault_type": match.fault_type,
        "confidence": match.confidence,
        "lead_time": match.lead_time,
        "matching_indicators": match.matching_indicators,
        "description": match.pattern_description,
    }


@app.post("/api/fingerprints/compare-to-healthy")
async def fingerprints_compare_healthy(data: dict):
    """
    Compare current metrics to healthy baseline.

    Input:
    {
        "domain": "pump",
        "metrics": {"coherence": 0.18, "lyapunov": -0.03}
    }

    Output:
    {
        "comparisons": {
            "coherence": {"deviation": -3.2, "status": "critical"},
            "lyapunov": {"deviation": 1.5, "status": "normal"}
        }
    }
    """
    domain = data.get("domain")
    metrics = data.get("metrics", {})

    if not domain:
        raise HTTPException(status_code=400, detail="Missing 'domain'")

    fps = get_fingerprint_service()
    fps.load_fingerprints(domain)

    comparison = fps.compare_to_healthy(metrics, domain)

    return {
        "comparisons": {
            metric: {"deviation": z, "status": status}
            for metric, (z, status) in comparison.items()
        }
    }


@app.post("/api/fingerprints/generate-from-tuning")
async def fingerprints_generate(data: dict):
    """
    Generate fingerprints from tuning results.

    After running tuning analysis, use this to create fingerprint files
    that can be used for production monitoring.

    Input:
    {
        "data_dir": "/path/to/results",
        "system_id": "pump_station_7",
        "domain": "pump",
        "fault_types": ["valve", "cavitation"]  # Optional, auto-detect if not provided
    }

    Output:
    {
        "generated": ["healthy_baseline.yaml", "deviation_valve.yaml", ...],
        "fingerprints_dir": "/path/to/fingerprints"
    }
    """
    global _last_results_path

    data_dir = data.get("data_dir")
    if not data_dir and _last_results_path:
        data_dir = str(_last_results_path)

    if not data_dir:
        raise HTTPException(status_code=400, detail="No data directory specified")

    system_id = data.get("system_id", "system")
    domain = data.get("domain", "custom")
    fault_types = data.get("fault_types")

    try:
        # Initialize services
        tuner = TuningService(data_dir, data.get("labels_path"))
        tuner._load_data()

        fps = get_fingerprint_service()
        generated = []

        # Generate healthy fingerprint
        healthy_fp = generate_healthy_fingerprint(
            tuner, system_id, domain,
            f"Baseline from {data_dir}"
        )
        path = fps.save_fingerprint(healthy_fp, domain, "healthy_baseline")
        generated.append(str(path))

        # Get fault types if not provided
        if not fault_types:
            try:
                label_df = tuner.get_label_summary()
                fault_types = label_df["label_name"].to_list() if label_df.height > 0 else []
            except Exception:
                fault_types = []

        # Generate deviation fingerprints
        for fault_type in fault_types:
            try:
                dev_fp = generate_deviation_fingerprint(
                    tuner, system_id, domain, fault_type,
                    f"Tuning analysis of {fault_type}"
                )
                path = fps.save_fingerprint(dev_fp, domain, f"deviation_{fault_type}")
                generated.append(str(path))
            except Exception as e:
                print(f"Warning: Could not generate fingerprint for {fault_type}: {e}")

        return {
            "generated": generated,
            "fingerprints_dir": str(fps.fingerprints_dir / domain),
        }

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/fingerprints/save")
async def fingerprints_save(data: dict):
    """
    Save a fingerprint to disk.

    Input:
    {
        "domain": "pump",
        "filename": "deviation_valve",
        "fingerprint": {
            "fingerprint_type": "deviation",
            "system_id": "pump_7",
            ...
        }
    }
    """
    domain = data.get("domain")
    filename = data.get("filename")
    fingerprint_data = data.get("fingerprint")

    if not domain or not filename or not fingerprint_data:
        raise HTTPException(status_code=400, detail="Missing domain, filename, or fingerprint")

    from prime.services.fingerprint_service import (
        HealthyFingerprint,
        DeviationFingerprint,
        FailureFingerprint,
    )

    fps = get_fingerprint_service()

    fp_type = fingerprint_data.get("fingerprint_type")
    if fp_type == "healthy":
        fp = HealthyFingerprint.from_dict(fingerprint_data)
    elif fp_type == "deviation":
        fp = DeviationFingerprint.from_dict(fingerprint_data)
    elif fp_type == "failure":
        fp = FailureFingerprint.from_dict(fingerprint_data)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown fingerprint type: {fp_type}")

    path = fps.save_fingerprint(fp, domain, filename)

    return {
        "saved": True,
        "path": str(path),
    }


def main():
    """CLI entry point for running the server."""
    import uvicorn
    uvicorn.run("prime.api:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
