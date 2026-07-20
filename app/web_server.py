import logging
import os
import re
import sys
import uuid
import boto3

import uvicorn
import secrets
from fastapi import FastAPI, HTTPException, Request, Depends, status
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
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

security = HTTPBasic(auto_error=False)

def get_current_username(credentials: HTTPBasicCredentials = Depends(security)):
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
            headers={"WWW-Authenticate": 'Basic realm="Political Discourse Analyzer"'},
        )
        
    admin_user = os.environ.get("ADMIN_USERNAME")
    admin_pass = os.environ.get("ADMIN_PASSWORD")
    
    if not admin_user or not admin_pass:
        logger.error("ADMIN_USERNAME or ADMIN_PASSWORD not set in environment.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication not configured on server",
        )

    correct_username = secrets.compare_digest(credentials.username, admin_user)
    correct_password = secrets.compare_digest(credentials.password, admin_pass)
    
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": 'Basic realm="Political Discourse Analyzer"'},
        )
    return credentials.username

app = FastAPI(title="Political Discourse Analyzer Dashboard", dependencies=[Depends(get_current_username)])

# DynamoDB Configuration
DYNAMODB_TABLE_NAME = os.environ.get("DYNAMODB_TABLE_NAME", "analyses")
dynamodb = boto3.resource("dynamodb", region_name=os.environ.get("AWS_REGION", "us-east-1"))
analyses_table = dynamodb.Table(DYNAMODB_TABLE_NAME)

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
        # Strip markdown JSON wrappers if present
        cleaned_str = re.sub(
            r"^```json\s*|\s*```$", "", output_raw.strip(), flags=re.MULTILINE
        )
        # Try to parse as JSON first
        try:
            parsed = json.loads(cleaned_str)
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

    logger.info(f"Starting SSE analysis for session {session_id}...")

    # Initialize ADK runner and session service
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        state={"user_input": input_text, "raw_input_text": input_text},
    )
    runner = Runner(
        agent=root_agent, app_name=app_name, session_service=session_service
    )

    async def event_generator():
        import json

        try:
            # Yield start event
            yield f"data: {json.dumps({'event': 'start', 'session_id': session_id})}\n\n"

            # Run agent programmatically with the user input
            async for event in runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=types.Content(
                    role="user", parts=[types.Part.from_text(text=input_text)]
                ),
            ):
                # Retrieve the active node name from path if running inside a workflow
                node_name = event.author
                if event.node_info and event.node_info.path:
                    node_name = event.node_info.path.split("/")[-1].split("@")[0]

                if node_name and event.is_final_response():
                    logger.info(f"[{node_name}] Event generated (SSE)")

                    # Fetch session state for the partial result
                    session = await session_service.get_session(
                        app_name=app_name, user_id=user_id, session_id=session_id
                    )
                    state = session.state

                    event_type = None
                    payload_data = {}

                    if node_name == "tag_article":
                        event_type = "input_processed"
                        payload_data = {"text": state.get("article_text", "")}
                    elif node_name == "ScopeClassifier":
                        event_type = "input_classified"
                        classification = state.get("scope_classification", {})
                        if isinstance(classification, str):
                            classification = {}
                        payload_data = {
                            "is_out_of_scope": classification.get(
                                "is_out_of_scope", False
                            ),
                            "is_satire": classification.get("is_satire", False),
                        }
                    elif node_name == "SecurityAuditor":
                        event_type = "security_completed"
                        security = state.get("security_evaluation", {})
                        if isinstance(security, str):
                            security = {}
                        payload_data = {
                            "is_safe": security.get("is_safe", True),
                            "risk_score": security.get("risk_score", 0),
                            "reason": security.get("reason", "Input is safe"),
                        }
                    elif node_name == "ParetoAnalyst":
                        event_type = "pareto_completed"
                        pareto_raw = state.get("pareto_analysis", "")
                        pareto_md, pareto_metrics = parse_analyst_output(pareto_raw)
                        payload_data = {
                            "analysis": pareto_md,
                            "metrics": pareto_metrics,
                        }
                    elif node_name == "SowellAnalyst":
                        event_type = "sowell_completed"
                        sowell_raw = state.get("sowell_analysis", "")
                        sowell_md, sowell_metrics = parse_analyst_output(sowell_raw)
                        payload_data = {
                            "analysis": sowell_md,
                            "metrics": sowell_metrics,
                        }
                    elif node_name == "MassPsychAnalyst":
                        event_type = "mass_psych_completed"
                        mass_psych_raw = state.get("mass_psych_analysis", "")
                        mass_psych_md, mass_psych_metrics = parse_analyst_output(
                            mass_psych_raw
                        )
                        payload_data = {
                            "analysis": mass_psych_md,
                            "metrics": mass_psych_metrics,
                        }
                    elif node_name == "FoucaultAnalyst":
                        event_type = "foucault_completed"
                        foucault_raw = state.get("foucault_analysis", "")
                        foucault_md, foucault_metrics = parse_analyst_output(
                            foucault_raw
                        )
                        payload_data = {
                            "analysis": foucault_md,
                            "metrics": foucault_metrics,
                        }
                    elif node_name == "GroundingEvaluator":
                        event_type = "grounding_completed"
                        evaluation = state.get("grounding_evaluation", {})
                        if isinstance(evaluation, str):
                            evaluation = {}
                        payload_data = {
                            "is_grounded": evaluation.get("is_grounded", True),
                            "grounding_score": evaluation.get("grounding_score", 100),
                            "feedback": evaluation.get("feedback", ""),
                            "hallucinated_elements": evaluation.get(
                                "hallucinated_elements", []
                            ),
                        }
                    elif node_name == "Synthesizer":
                        event_type = "synthesis_completed"
                        final_report = state.get(
                            "final_report", "No final report generated."
                        )

                        # Programmatically parse title, summary, and report_md from final_report
                        title = "Discourse Analysis"
                        summary = ""
                        report_md = "No final report generated."

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
                                logger.error(
                                    f"Failed to dump Pydantic model in SSE: {e}"
                                )
                        elif isinstance(final_report, str):
                            cleaned_report = re.sub(
                                r"^```json\s*|\s*```$",
                                "",
                                final_report.strip(),
                                flags=re.MULTILINE,
                            )
                            try:
                                report_data = json.loads(cleaned_report)
                                title = report_data.get("title", title)
                                summary = report_data.get("summary", summary)
                                report_md = report_data.get("report_md", report_md)
                            except Exception as e:
                                logger.info(
                                    f"Synthesizer output is not JSON, treating as raw markdown: {e}"
                                )
                                report_md = final_report
                                lines = final_report.strip().split("\n")
                                for line in lines:
                                    if line.startswith("# "):
                                        title = (
                                            line[2:]
                                            .strip()
                                            .replace("*", "")
                                            .replace("_", "")
                                        )
                                        break

                                lower_report = final_report.lower()
                                summary_idx = -1
                                for keyword in ["executive summary", "summary"]:
                                    summary_idx = lower_report.find(keyword)
                                    if summary_idx != -1:
                                        summary_text = final_report[
                                            summary_idx + len(keyword) :
                                        ].strip()
                                        summary_text = summary_text.lstrip(":- \t\r\n")
                                        end_idx = len(summary_text)
                                        for term in ["\n\n", "\n#", "\n*"]:
                                            pos = summary_text.find(term)
                                            if pos != -1 and pos < end_idx:
                                                end_idx = pos
                                        summary = summary_text[:end_idx].strip()
                                        break
                                if not summary:
                                    content_lines = [
                                        line.strip()
                                        for line in lines
                                        if line.strip()
                                        and not line.startswith(("#", ">", "*", "-"))
                                    ]
                                    if content_lines:
                                        summary = " ".join(content_lines[:2])
                                        if len(summary) > 200:
                                            summary = summary[:197] + "..."
                        else:
                            report_md = str(final_report)

                        payload_data = {
                            "title": title,
                            "summary": summary,
                            "report_md": report_md,
                        }

                    if event_type:
                        sse_msg = {
                            "event": event_type,
                            "session_id": session_id,
                            "data": payload_data,
                        }
                        yield f"data: {json.dumps(sse_msg)}\n\n"

            # Finally, retrieve full final state to save to SQLite database
            session = await session_service.get_session(
                app_name=app_name, user_id=user_id, session_id=session_id
            )
            state = session.state

            final_report = state.get("final_report", "No final report generated.")
            original_text = state.get("article_text", input_text)

            pareto_raw = state.get("pareto_analysis", "")
            sowell_raw = state.get("sowell_analysis", "")
            mass_psych_raw = state.get("mass_psych_analysis", "")
            foucault_raw = state.get("foucault_analysis", "")

            # Parsing to extract title/summary for db record
            title = "Discourse Analysis"
            summary = ""
            report_md = "No final report generated."

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
                except Exception:
                    pass
            elif isinstance(final_report, str):
                cleaned_report = re.sub(
                    r"^```json\s*|\s*```$", "", final_report.strip(), flags=re.MULTILINE
                )
                try:
                    report_data = json.loads(cleaned_report)
                    title = report_data.get("title", title)
                    summary = report_data.get("summary", summary)
                    report_md = report_data.get("report_md", report_md)
                except Exception:
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
                            summary_text = final_report[
                                summary_idx + len(keyword) :
                            ].strip()
                            summary_text = summary_text.lstrip(":- \t\r\n")
                            end_idx = len(summary_text)
                            for term in ["\n\n", "\n#", "\n*"]:
                                pos = summary_text.find(term)
                                if pos != -1 and pos < end_idx:
                                    end_idx = pos
                            summary = summary_text[:end_idx].strip()
                            break
                    if not summary:
                        content_lines = [
                            line.strip()
                            for line in lines
                            if line.strip()
                            and not line.startswith(("#", ">", "*", "-"))
                        ]
                        if content_lines:
                            summary = " ".join(content_lines[:2])
                            if len(summary) > 200:
                                summary = summary[:197] + "..."

            db_id = str(uuid.uuid4())
            try:
                import datetime
                created_at = datetime.datetime.utcnow().isoformat()
                item = {
                    "id": db_id,
                    "created_at": created_at,
                    "title": title,
                    "source": input_text,
                    "report_md": report_md,
                    "summary": summary,
                    "original_text": original_text,
                    "pareto_analysis": get_db_value(pareto_raw),
                    "sowell_analysis": get_db_value(sowell_raw),
                    "mass_psych_analysis": get_db_value(mass_psych_raw),
                    "foucault_analysis": get_db_value(foucault_raw),
                }
                analyses_table.put_item(Item=item)
                logger.info(
                    f"Successfully saved analysis report {db_id} to DynamoDB."
                )
            except Exception as e:
                logger.error(f"Failed to save analysis report to DynamoDB: {e}")

            # Yield complete event with database row ID
            yield f"data: {json.dumps({'event': 'complete', 'session_id': session_id, 'db_id': db_id})}\n\n"

        except Exception as e:
            logger.error(f"Error during SSE analysis generator: {e}", exc_info=True)
            yield f"data: {json.dumps({'event': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/history")
async def get_history():
    try:
        response = analyses_table.scan(
            ProjectionExpression="id, title, #src, summary, created_at",
            ExpressionAttributeNames={"#src": "source"}
        )
        items = response.get('Items', [])
        items.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        return JSONResponse(content=items)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/history/{report_id}")
async def get_report(report_id: str):
    try:
        response = analyses_table.get_item(Key={"id": report_id})
        item = response.get('Item')

        if not item:
            raise HTTPException(status_code=404, detail="Report not found")

        pareto_db = item.get("pareto_analysis", "")
        sowell_db = item.get("sowell_analysis", "")
        mass_psych_db = item.get("mass_psych_analysis", "")
        foucault_db = item.get("foucault_analysis", "")

        pareto_md, pareto_metrics = parse_analyst_output(pareto_db)
        sowell_md, sowell_metrics = parse_analyst_output(sowell_db)
        mass_psych_md, mass_psych_metrics = parse_analyst_output(mass_psych_db)
        foucault_md, foucault_metrics = parse_analyst_output(foucault_db)

        report = {
            "id": item.get("id"),
            "title": item.get("title"),
            "source": item.get("source"),
            "report_md": item.get("report_md"),
            "summary": item.get("summary"),
            "created_at": item.get("created_at"),
            "original_text": item.get("original_text", ""),
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
async def delete_report(report_id: str):
    try:
        analyses_table.delete_item(Key={"id": report_id})
        return JSONResponse(
            content={"status": "success", "message": f"Report {report_id} deleted."}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


if __name__ == "__main__":
    # Run the web server
    uvicorn.run("app.web_server:app", host="127.0.0.1", port=8000, reload=True)
