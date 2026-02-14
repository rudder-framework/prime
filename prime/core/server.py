"""
RUDDER Server
Serves static files + provides API endpoints (including LLM unit suggestions).

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python -m prime.server

Or:
    uvicorn prime.server:app --reload --port 8000
"""

import os
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI(title="RUDDER", description="Signal Analysis Engine")

# Static files directory (in explorer module)
STATIC_DIR = Path(__file__).parent.parent / "explorer" / "static"


# ============================================================================
# API Models
# ============================================================================

class UnitSuggestionRequest(BaseModel):
    signals: list[str]


class UnitSuggestionResponse(BaseModel):
    suggestions: dict[str, str]


# ============================================================================
# API Endpoints
# ============================================================================

@app.post("/api/suggest-units")
async def suggest_units(request: UnitSuggestionRequest) -> dict:
    """
    Use Anthropic Claude to suggest units for signal names.
    Requires ANTHROPIC_API_KEY environment variable.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="ANTHROPIC_API_KEY not set. Run: export ANTHROPIC_API_KEY=sk-ant-..."
        )

    try:
        import anthropic
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="anthropic package not installed. Run: pip install anthropic"
        )

    # Build prompt
    signal_list = "\n".join(f"- {s}" for s in request.signals)

    prompt = f"""You are an expert in industrial process signals and sensor data.

Given these signal names from a dataset, suggest the most likely unit for each one.
Return ONLY a JSON object mapping signal names to units. Use standard abbreviations.

Common units to consider:
- Pressure: PSI, kPa, bar, Pa
- Temperature: degC, degF, K
- Flow: gpm, lpm, m3/s, kg/s
- Velocity: m/s, ft/s, rpm
- Electrical: V, A, kW, MW, Hz
- Other: %, ppm, kg, m, mm

Signal names:
{signal_list}

Respond with ONLY valid JSON, no explanation. Example:
{{"signal_name": "PSI", "another_signal": "degC"}}"""

    client = anthropic.Anthropic(api_key=api_key)

    try:
        message = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )

        # Parse response
        response_text = message.content[0].text.strip()

        # Handle markdown code blocks
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            response_text = "\n".join(lines[1:-1])

        suggestions = json.loads(response_text)
        return suggestions

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse LLM response: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "anthropic_key_set": bool(os.environ.get("ANTHROPIC_API_KEY"))
    }


@app.get("/api/ai/status")
async def ai_status():
    """Check if AI assistance is available."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    return {
        "available": bool(api_key),
        "message": "Ready" if api_key else "ANTHROPIC_API_KEY not set"
    }


@app.get("/api/prism/status")
async def prism_status():
    """Check PRISM server status."""
    prism_url = os.environ.get("PRISM_URL", "http://localhost:8001")
    try:
        import httpx
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{prism_url}/health")
            if resp.status_code == 200:
                return {"available": True, "status": "Connected", "url": prism_url}
    except Exception:
        pass
    return {"available": False, "status": "Not connected", "url": prism_url}


@app.get("/api/load-parquets")
async def load_parquets(path: str):
    """
    List parquet files in a directory for loading in the dashboard.
    Returns paths that the browser can use with DuckDB-WASM.
    """
    from pathlib import Path as P

    dir_path = P(path)
    if not dir_path.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {path}")

    if not dir_path.is_dir():
        raise HTTPException(status_code=400, detail=f"Not a directory: {path}")

    # Expected parquet files
    expected = [
        'observations.parquet',
        'vector.parquet',
        'geometry.parquet',
        'physics.parquet',
        'dynamics.parquet',
        'topology.parquet',
        'information_flow.parquet',
    ]

    files = {}
    for name in expected:
        file_path = dir_path / name
        if file_path.exists():
            # Return the table name (without .parquet) and full path
            table_name = name.replace('.parquet', '')
            files[table_name] = str(file_path)

    if not files:
        raise HTTPException(status_code=404, detail=f"No parquet files found in: {path}")

    return {"files": files, "path": path}


class AIInterpretRequest(BaseModel):
    tables: list[str]
    context: str = ""


# ============================================================================
# Schema Transformation API
# ============================================================================

class ColumnMapping(BaseModel):
    """How to map a source column."""
    role: str  # 'signal', 'index', 'cohort', 'ignore'
    name: str | None = None  # Optional rename


class TransformRequest(BaseModel):
    """Request to transform data to observations.parquet format."""
    mappings: dict[str, ColumnMapping]  # source_column -> mapping


@app.post("/api/schema/preview")
async def schema_preview(file: bytes = None):
    """
    Upload a file and get column info for mapping.
    Accepts CSV or Parquet via multipart form.
    """
    # This will be handled by the upload endpoint below
    pass


from fastapi import UploadFile, File, Form
from fastapi.responses import StreamingResponse
import tempfile
import io


@app.post("/api/schema/detect")
async def schema_detect(file: UploadFile = File(...)):
    """
    Upload a file and detect its columns/types.
    Returns column info for the UI to display.
    """
    import polars as pl

    content = await file.read()
    filename = file.filename or "data"

    try:
        # Detect format and read
        if filename.endswith('.parquet'):
            df = pl.read_parquet(io.BytesIO(content))
        elif filename.endswith('.csv') or filename.endswith('.txt'):
            df = pl.read_csv(io.BytesIO(content), infer_schema_length=1000)
        elif filename.endswith('.tsv'):
            df = pl.read_csv(io.BytesIO(content), separator='\t', infer_schema_length=1000)
        else:
            # Try CSV as default
            df = pl.read_csv(io.BytesIO(content), infer_schema_length=1000)

        # Get column info
        columns = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            sample_values = df[col].head(5).to_list()
            n_unique = df[col].n_unique()
            n_null = df[col].null_count()

            # Auto-suggest role based on column name and data
            suggested_role = 'signal'  # default
            lower = col.lower()

            if any(x in lower for x in ['time', 'date', 'timestamp', 'index', 'row']):
                suggested_role = 'index'
            elif any(x in lower for x in ['id', 'entity', 'unit', 'tag', 'name', 'cohort', 'group']):
                suggested_role = 'cohort'
            elif dtype in ('String', 'Utf8', 'Categorical'):
                # String columns are usually metadata, not signals
                suggested_role = 'cohort'
            elif n_unique <= 1:
                suggested_role = 'ignore'  # Constant column

            columns.append({
                'name': col,
                'dtype': dtype,
                'sample': sample_values[:3],
                'n_unique': n_unique,
                'n_null': n_null,
                'suggested_role': suggested_role,
            })

        return {
            'filename': filename,
            'n_rows': len(df),
            'n_cols': len(df.columns),
            'columns': columns,
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")


@app.post("/api/schema/transform")
async def schema_transform(
    file: UploadFile = File(...),
    mappings: str = Form(...),  # JSON string of mappings
):
    """
    Transform uploaded data to observations.parquet format.

    Mappings JSON format:
    {
        "column_name": {"role": "signal|index|cohort|ignore", "rename": "optional_new_name"},
        ...
    }

    Returns the transformed parquet file.
    """
    import polars as pl
    import json

    content = await file.read()
    filename = file.filename or "data"
    mapping_dict = json.loads(mappings)

    try:
        # Read file
        if filename.endswith('.parquet'):
            df = pl.read_parquet(io.BytesIO(content))
        elif filename.endswith('.tsv'):
            df = pl.read_csv(io.BytesIO(content), separator='\t', infer_schema_length=10000)
        else:
            df = pl.read_csv(io.BytesIO(content), infer_schema_length=10000)

        # Identify columns by role
        signal_cols = []
        index_col = None
        cohort_col = None

        for col, mapping in mapping_dict.items():
            if col not in df.columns:
                continue
            role = mapping.get('role', 'ignore')
            if role == 'signal':
                signal_cols.append(col)
            elif role == 'index':
                index_col = col
            elif role == 'cohort':
                cohort_col = col

        if not signal_cols:
            raise HTTPException(status_code=400, detail="No signal columns selected")

        # Build the long-format dataframe
        # Select only signal columns and add row index
        base_df = df.select(signal_cols).with_row_index('_row_idx')

        # Unpivot (melt) from wide to long - only signal columns become values
        observations = base_df.unpivot(
            index='_row_idx',
            variable_name='signal_id',
            value_name='value',
        )

        # Create proper I column (sequential per signal_id)
        observations = (
            observations
            .sort(['signal_id', '_row_idx'])
            .with_columns([
                pl.col('_row_idx').cast(pl.UInt32).alias('I'),
                pl.col('value').cast(pl.Float64),
            ])
            .drop('_row_idx')
        )

        # Add cohort if present
        if cohort_col:
            # Get unique cohort value (assuming one cohort per file)
            cohort_values = df[cohort_col].unique().to_list()
            if len(cohort_values) == 1:
                cohort_name = str(cohort_values[0])
            else:
                # Multiple cohorts - use filename as cohort
                cohort_name = filename.rsplit('.', 1)[0]

            observations = observations.with_columns(
                pl.lit(cohort_name).alias('cohort')
            )

        # Reorder columns to standard order
        col_order = ['signal_id', 'I', 'value']
        if 'cohort' in observations.columns:
            col_order = ['cohort'] + col_order
        observations = observations.select(col_order)

        # Write to buffer
        buffer = io.BytesIO()
        observations.write_parquet(buffer)
        buffer.seek(0)

        return StreamingResponse(
            buffer,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f"attachment; filename=observations.parquet"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transform failed: {e}")


@app.post("/api/schema/validate")
async def schema_validate(file: UploadFile = File(...)):
    """
    Validate that a parquet file matches observations.parquet schema.
    """
    import polars as pl

    content = await file.read()

    try:
        df = pl.read_parquet(io.BytesIO(content))
    except Exception as e:
        return {
            'valid': False,
            'errors': [f"Cannot read parquet: {e}"],
            'warnings': [],
        }

    errors = []
    warnings = []

    # Required columns
    required = {'signal_id': 'String', 'I': 'UInt32', 'value': 'Float64'}
    optional = {'cohort': 'String'}

    for col, expected_type in required.items():
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")
        else:
            actual = str(df[col].dtype)
            if expected_type not in actual and actual not in expected_type:
                # Allow some flexibility (e.g., Int64 for I is ok-ish)
                if col == 'I' and 'Int' in actual:
                    warnings.append(f"Column 'I' is {actual}, expected UInt32 (will work but not ideal)")
                elif col == 'value' and 'Float' in actual:
                    pass  # Float32 is fine
                else:
                    errors.append(f"Column '{col}' is {actual}, expected {expected_type}")

    # Check I is sequential per signal_id
    if 'signal_id' in df.columns and 'I' in df.columns:
        for sig in df['signal_id'].unique().to_list()[:5]:  # Check first 5 signals
            sig_data = df.filter(pl.col('signal_id') == sig).sort('I')
            i_vals = sig_data['I'].to_list()
            expected = list(range(len(i_vals)))
            if i_vals != expected:
                warnings.append(f"Signal '{sig}': I not sequential (starts at {i_vals[0] if i_vals else '?'})")
                break

    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings,
        'schema': {col: str(df[col].dtype) for col in df.columns},
        'n_rows': len(df),
        'n_signals': df['signal_id'].n_unique() if 'signal_id' in df.columns else 0,
    }


@app.post("/api/ai/interpret")
async def ai_interpret(request: AIInterpretRequest):
    """
    Generate AI interpretation of analysis results.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="ANTHROPIC_API_KEY not set"
        )

    try:
        import anthropic
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="anthropic package not installed"
        )

    tables_str = ", ".join(request.tables)

    prompt = f"""You are an expert in structural health monitoring and time series analysis.

The user has loaded the following data tables: {tables_str}

These come from the RUDDER/PRISM four-pillar analysis system:
- Geometry: Eigenvalue coherence, effective dimension, signal coupling
- Dynamics: Lyapunov exponents, RQA metrics (determinism, laminarity)
- Topology: Betti numbers, persistence homology, attractor shape
- Information: Transfer entropy, causal hierarchy, feedback loops

Based on the available data, provide a brief interpretation of what analyses are available
and what insights they can provide. Keep it concise (2-3 paragraphs).

{f'Additional context: {request.context}' if request.context else ''}"""

    client = anthropic.Anthropic(api_key=api_key)

    try:
        message = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )

        return {"interpretation": message.content[0].text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI error: {e}")


# ============================================================================
# Markdown Generation
# ============================================================================

class MarkdownRequest(BaseModel):
    documents: list[str]


@app.post("/api/generate-markdowns")
async def generate_markdowns(request: MarkdownRequest):
    """
    Generate markdown documentation from SQL files.

    Creates .md files in rudder/sql/docs/ with SQL syntax highlighting.
    """
    sql_dir = Path(__file__).parent / "sql"
    docs_dir = sql_dir / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    generated = []
    errors = []

    for doc_id in request.documents:
        sql_file = sql_dir / f"{doc_id}.sql"
        md_file = docs_dir / f"{doc_id}.md"

        if not sql_file.exists():
            errors.append(f"{doc_id}.sql not found")
            continue

        try:
            sql_content = sql_file.read_text()
            md_content = f"# {doc_id}\n\n```sql\n{sql_content}\n```\n"
            md_file.write_text(md_content)
            generated.append(f"{doc_id}.md")
        except Exception as e:
            errors.append(f"{doc_id}: {str(e)}")

    return {
        "status": "complete" if not errors else "partial",
        "files": generated,
        "errors": errors,
        "output_dir": str(docs_dir)
    }


@app.get("/api/sql-docs")
async def list_sql_docs():
    """List available SQL documentation files."""
    sql_dir = Path(__file__).parent / "sql"
    docs_dir = sql_dir / "docs"

    sql_files = sorted(sql_dir.glob("[0-9]*.sql"))

    docs = []
    for f in sql_files:
        doc_id = f.stem
        md_exists = (docs_dir / f"{doc_id}.md").exists()
        docs.append({
            "id": doc_id,
            "sql_file": f.name,
            "md_exists": md_exists
        })

    return {"documents": docs}


# ============================================================================
# Static Files (must be last)
# ============================================================================

@app.get("/")
async def root():
    """Serve index.html"""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/v2")
async def dashboard_v2():
    """Serve the new restructured dashboard (index_v2.html)"""
    return FileResponse(STATIC_DIR / "index_v2.html")


@app.get("/wizard")
async def wizard():
    """Serve the data transformation wizard"""
    return FileResponse(STATIC_DIR / "wizard.html")


# Mount static files for everything else
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    api_status = '✓ Set' if os.environ.get('ANTHROPIC_API_KEY') else '✗ Not set'
    print(f"""
╔═══════════════════════════════════════════════════════════╗
║  RUDDER Server                                            ║
╠═══════════════════════════════════════════════════════════╣
║  Dashboard:  http://localhost:{port}                        ║
║  New UI:     http://localhost:{port}/v2                     ║
║                                                           ║
║  AI Assist:  {api_status:<43} ║
╚═══════════════════════════════════════════════════════════╝
""")

    uvicorn.run(app, host="0.0.0.0", port=port)
