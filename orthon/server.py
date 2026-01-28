"""
ORTHON Server
Serves static files + provides API endpoints (including LLM unit suggestions).

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python -m orthon.server

Or:
    uvicorn orthon.server:app --reload --port 8000
"""

import os
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI(title="ORTHON", description="Signal Analysis Engine")

# Static files directory
STATIC_DIR = Path(__file__).parent / "static"


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


# ============================================================================
# Markdown Generation
# ============================================================================

class MarkdownRequest(BaseModel):
    documents: list[str]


@app.post("/api/generate-markdowns")
async def generate_markdowns(request: MarkdownRequest):
    """
    Generate markdown documentation from SQL files.

    Creates .md files in orthon/sql/docs/ with SQL syntax highlighting.
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


# Mount static files for everything else
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    print(f"""
╔═══════════════════════════════════════════════════════════╗
║  ORTHON Server                                            ║
╠═══════════════════════════════════════════════════════════╣
║  http://localhost:{port}                                    ║
║                                                           ║
║  API Key: {'✓ Set' if os.environ.get('ANTHROPIC_API_KEY') else '✗ Not set (export ANTHROPIC_API_KEY=...)'}                              ║
╚═══════════════════════════════════════════════════════════╝
""")

    uvicorn.run(app, host="0.0.0.0", port=port)
