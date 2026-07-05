import logging
import os
import sqlite3
import sys
import uuid

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# Ensure the parent directory is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from app.agent import root_agent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("discourse_anal_web")

app = FastAPI(title="Political Discourse Analyzer Dashboard")

# SQLite DB Path (Shared with MCP Server)
DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "analyses.db"
)

# Jinja2 Templates setup
TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
os.makedirs(TEMPLATES_DIR, exist_ok=True)
templates = Jinja2Templates(directory=TEMPLATES_DIR)


class AnalysisRequest(BaseModel):
    input_text: str


def parse_analyst_output(output_raw):
    """
    Safely parses the analyst's output, which could be a parsed dictionary,
    a Pydantic object, a JSON string, or legacy plain text markdown.
    Returns a tuple of (analysis_md, metrics_dict).
    """
    import json
    analysis_md = str(output_raw)
    metrics = {}

    if isinstance(output_raw, dict):
        analysis_md = output_raw.get("analysis_md", analysis_md)
        metrics = {k: v for k, v in output_raw.items() if k != "analysis_md"}
    elif hasattr(output_raw, "model_dump"):
        try:
            output_dict = output_raw.model_dump()
            analysis_md = output_dict.get("analysis_md", analysis_md)
            metrics = {k: v for k, v in output_dict.items() if k != "analysis_md"}
        except Exception as e:
            logger.error(f"Failed to dump analyst Pydantic model: {e}")
    elif isinstance(output_raw, str) and output_raw.strip():
        # Try to parse as JSON first
        try:
            parsed = json.loads(output_raw)
            if isinstance(parsed, dict):
                analysis_md = parsed.get("analysis_md", analysis_md)
                metrics = {k: v for k, v in parsed.items() if k != "analysis_md"}
        except Exception:
            # Not JSON, treat as raw markdown
            pass
    return analysis_md, metrics


def get_db_value(raw_val):
    """
    Serializes raw values to a JSON string if they are dicts or Pydantic models.
    Otherwise, returns the string representation as is.
    """
    import json
    if isinstance(raw_val, dict):
        return json.dumps(raw_val)
    elif hasattr(raw_val, "model_dump"):
        try:
            return json.dumps(raw_val.model_dump())
        except Exception:
            pass
    return str(raw_val)


@app.get("/", response_class=HTMLResponse)
async def get_dashboard(request: Request):
    return templates.TemplateResponse(
        request=request, name="dashboard.html", context={"request": request}
    )


@app.post("/analyze")
async def analyze_discourse(payload: AnalysisRequest):
    input_text = payload.input_text.strip()
    if not input_text:
        raise HTTPException(status_code=400, detail="Input text or URL cannot be empty")

    user_id = "web_user"
    session_id = str(uuid.uuid4())
    app_name = "app"

    logger.info(f"Starting analysis for session {session_id}...")

    # Initialize ADK runner and session service
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name=app_name, user_id=user_id, session_id=session_id
    )
    runner = Runner(
        agent=root_agent, app_name=app_name, session_service=session_service
    )

    try:
        # Run agent programmatically with the user input
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=types.Content(
                role="user", parts=[types.Part.from_text(text=input_text)]
            ),
        ):
            # Log sub-agent execution status in the console
            if event.author:
                logger.info(f"[{event.author}] Event generated")

        # Retrieve the accumulated state containing individual sub-agent analyses
        session = await session_service.get_session(
            app_name=app_name, user_id=user_id, session_id=session_id
        )
        state = session.state

        # Build payload response
        final_report = state.get("final_report", "No final report generated.")
        original_text = state.get("article_text", input_text)
        
        pareto_raw = state.get("pareto_analysis", "")
        sowell_raw = state.get("sowell_analysis", "")
        mass_psych_raw = state.get("mass_psych_analysis", "")
        foucault_raw = state.get("foucault_analysis", "")

        pareto_md, pareto_metrics = parse_analyst_output(pareto_raw)
        sowell_md, sowell_metrics = parse_analyst_output(sowell_raw)
        mass_psych_md, mass_psych_metrics = parse_analyst_output(mass_psych_raw)
        foucault_md, foucault_metrics = parse_analyst_output(foucault_raw)

        # Programmatically parse title, summary, and report_md from final_report
        import json
        title = "Discourse Analysis"
        summary = ""
        report_md = "No final report generated."

        # Support dict (automatically parsed by ADK), Pydantic objects, and raw strings
        if isinstance(final_report, dict):
            title = final_report.get("title", title)
            summary = final_report.get("summary", summary)
            report_md = final_report.get("report_md", report_md)
        elif hasattr(final_report, "model_dump"):
            try:
                report_dict = final_report.model_dump()
                title = report_dict.get("title", title)
                summary = report_dict.get("summary", summary)
                report_md = report_dict.get("report_md", report_md)
            except Exception as e:
                logger.error(f"Failed to dump Pydantic model: {e}")
        elif isinstance(final_report, str):
            try:
                # Attempt to parse final_report as a JSON object
                report_data = json.loads(final_report)
                title = report_data.get("title", title)
                summary = report_data.get("summary", summary)
                report_md = report_data.get("report_md", report_md)
            except Exception as e:
                logger.info(f"Synthesizer output is not JSON, treating as raw markdown: {e}")
                # Fallback parsing logic for unstructured markdown output
                report_md = final_report
                lines = final_report.strip().split("\n")
                for line in lines:
                    if line.startswith("# "):
                        title = line[2:].strip().replace("*", "").replace("_", "")
                        break

                lower_report = final_report.lower()
                summary_idx = -1
                for keyword in ["executive summary", "summary"]:
                    summary_idx = lower_report.find(keyword)
                    if summary_idx != -1:
                        summary_text = final_report[summary_idx + len(keyword):].strip()
                        summary_text = summary_text.lstrip(":- \t\r\n")
                        end_idx = len(summary_text)
                        for term in ["\n\n", "\n#", "\n*"]:
                            pos = summary_text.find(term)
                            if pos != -1 and pos < end_idx:
                                end_idx = pos
                        summary = summary_text[:end_idx].strip()
                        break
                if not summary:
                    # Fallback: first two content lines
                    content_lines = [line.strip() for line in lines if line.strip() and not line.startswith(("#", ">", "*", "-"))]
                    if content_lines:
                        summary = " ".join(content_lines[:2])
                        if len(summary) > 200:
                            summary = summary[:197] + "..."
        else:
            logger.warning(f"Unexpected type for final_report: {type(final_report)}")
            report_md = str(final_report)

        # Save to SQLite database programmatically
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO analyses (title, source, report_md, summary, original_text, pareto_analysis, sowell_analysis, mass_psych_analysis, foucault_analysis) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    title,
                    input_text,
                    report_md,
                    summary,
                    original_text,
                    get_db_value(pareto_raw),
                    get_db_value(sowell_raw),
                    get_db_value(mass_psych_raw),
                    get_db_value(foucault_raw),
                ),
            )
            conn.commit()
            conn.close()
            logger.info("Successfully saved analysis report to SQLite database programmatically.")
        except Exception as e:
            logger.error(f"Failed to save analysis report to database: {e}")

        response_data = {
            "session_id": session_id,
            "article_text": original_text,
            "pareto_analysis": pareto_md,
            "sowell_analysis": sowell_md,
            "mass_psych_analysis": mass_psych_md,
            "foucault_analysis": foucault_md,
            "final_report": report_md,
            "pareto_metrics": pareto_metrics,
            "sowell_metrics": sowell_metrics,
            "mass_psych_metrics": mass_psych_metrics,
            "foucault_metrics": foucault_metrics,
        }

        return JSONResponse(content=response_data)

    except Exception as e:
        logger.error(f"Error during analysis: {e!s}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e!s}") from e


@app.get("/history")
async def get_history():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, title, source, summary, created_at FROM analyses ORDER BY id DESC"
        )
        rows = cursor.fetchall()
        conn.close()

        reports = []
        for row in rows:
            reports.append(
                {
                    "id": row[0],
                    "title": row[1],
                    "source": row[2],
                    "summary": row[3],
                    "created_at": row[4],
                }
            )
        return JSONResponse(content=reports)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/history/{report_id}")
async def get_report(report_id: int):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, title, source, report_md, summary, created_at, original_text, pareto_analysis, sowell_analysis, mass_psych_analysis, foucault_analysis FROM analyses WHERE id = ?",
            (report_id,),
        )
        row = cursor.fetchone()
        conn.close()

        if not row:
            raise HTTPException(status_code=404, detail="Report not found")

        pareto_db = row[7] or ""
        sowell_db = row[8] or ""
        mass_psych_db = row[9] or ""
        foucault_db = row[10] or ""

        pareto_md, pareto_metrics = parse_analyst_output(pareto_db)
        sowell_md, sowell_metrics = parse_analyst_output(sowell_db)
        mass_psych_md, mass_psych_metrics = parse_analyst_output(mass_psych_db)
        foucault_md, foucault_metrics = parse_analyst_output(foucault_db)

        report = {
            "id": row[0],
            "title": row[1],
            "source": row[2],
            "report_md": row[3],
            "summary": row[4],
            "created_at": row[5],
            "original_text": row[6] or "",
            "pareto_analysis": pareto_md,
            "sowell_analysis": sowell_md,
            "mass_psych_analysis": mass_psych_md,
            "foucault_analysis": foucault_md,
            "pareto_metrics": pareto_metrics,
            "sowell_metrics": sowell_metrics,
            "mass_psych_metrics": mass_psych_metrics,
            "foucault_metrics": foucault_metrics,
        }
        return JSONResponse(content=report)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.delete("/history/{report_id}")
async def delete_report(report_id: int):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM analyses WHERE id = ?", (report_id,))
        conn.commit()
        conn.close()
        return JSONResponse(content={"status": "success", "message": f"Report {report_id} deleted."})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


if __name__ == "__main__":
    # Ensure analyses.db is initialized before starting
    from app.mcp_server import init_db

    init_db()

    # Run the web server
    uvicorn.run("app.web_server:app", host="127.0.0.1", port=8000, reload=True)
