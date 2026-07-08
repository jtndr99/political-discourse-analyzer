import os
from pydantic import BaseModel

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.models import Gemini
from google.genai import types
from google.adk.workflow import Workflow, node, START
from google.adk.agents.context import Context

from app.mcp_client import get_mcp_toolset
from app.tools import fetch_web_page
from app.prompts import (
    INPUT_AGENT_INSTRUCTION,
    PARETO_ANALYST_INSTRUCTION,
    SOWELL_ANALYST_INSTRUCTION,
    MASS_PSYCH_ANALYST_INSTRUCTION,
    FOUCAULT_ANALYST_INSTRUCTION,
    SYNTHESIZER_INSTRUCTION,
    GROUNDING_EVALUATOR_INSTRUCTION,
    SECURITY_AUDITOR_INSTRUCTION,
)

class SynthesisReport(BaseModel):
    title: str
    subtitle: str
    summary: str
    report_md: str

class ParetoAnalysisSchema(BaseModel):
    analysis_md: str
    fox_drive_score: int
    derivations_detected: list[str]

class SowellAnalysisSchema(BaseModel):
    analysis_md: str
    unconstrained_score: int
    schmitt_intensity: int

class MassPsychAnalysisSchema(BaseModel):
    analysis_md: str
    mimetic_tension: int
    scapegoat_index: int

class FoucaultAnalysisSchema(BaseModel):
    analysis_md: str
    mechanisms_detected: list[str]

class GroundingEvaluationSchema(BaseModel):
    is_grounded: bool
    grounding_score: int
    feedback: str
    hallucinated_elements: list[str]

class SecurityEvaluationSchema(BaseModel):
    is_safe: bool
    risk_score: int
    reason: str

# Load local environment variables from .env
load_dotenv()

# Resolve Gemini API Configuration
if os.environ.get("GEMINI_API_KEY"):
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "False"
else:
    # Use Vertex AI defaults
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
    if "GOOGLE_CLOUD_PROJECT" not in os.environ:
        os.environ["GOOGLE_CLOUD_PROJECT"] = "YOUR_PROJECT_ID"
    if "GOOGLE_CLOUD_LOCATION" not in os.environ:
        os.environ["GOOGLE_CLOUD_LOCATION"] = "global"

# Define the shared MCP Toolset
mcp_toolset = get_mcp_toolset()

# Define model settings
MODEL_NAME = "gemini-3.1-flash-lite"
DEFAULT_GEN_CONFIG = types.GenerateContentConfig(temperature=0.0)
SYNTHESIZER_GEN_CONFIG = types.GenerateContentConfig(temperature=0.3)


def get_model():
    return Gemini(model=MODEL_NAME, retry_options=types.HttpRetryOptions(attempts=3))


def create_input_agent() -> Agent:
    return Agent(
        name="InputAgent",
        model=get_model(),
        instruction=INPUT_AGENT_INSTRUCTION,
        tools=[fetch_web_page],
        output_key="article_text",
        generate_content_config=DEFAULT_GEN_CONFIG,
    )


def create_security_auditor() -> Agent:
    return Agent(
        name="SecurityAuditor",
        model=get_model(),
        instruction=SECURITY_AUDITOR_INSTRUCTION,
        output_key="security_evaluation",
        output_schema=SecurityEvaluationSchema,
        generate_content_config=DEFAULT_GEN_CONFIG,
    )


def create_pareto_analyst() -> Agent:
    return Agent(
        name="ParetoAnalyst",
        model=get_model(),
        instruction=PARETO_ANALYST_INSTRUCTION,
        tools=[mcp_toolset],
        output_key="pareto_analysis",
        output_schema=ParetoAnalysisSchema,
        generate_content_config=DEFAULT_GEN_CONFIG,
    )


def create_sowell_analyst() -> Agent:
    return Agent(
        name="SowellAnalyst",
        model=get_model(),
        instruction=SOWELL_ANALYST_INSTRUCTION,
        tools=[mcp_toolset],
        output_key="sowell_analysis",
        output_schema=SowellAnalysisSchema,
        generate_content_config=DEFAULT_GEN_CONFIG,
    )


def create_mass_psych_analyst() -> Agent:
    return Agent(
        name="MassPsychAnalyst",
        model=get_model(),
        instruction=MASS_PSYCH_ANALYST_INSTRUCTION,
        tools=[mcp_toolset],
        output_key="mass_psych_analysis",
        output_schema=MassPsychAnalysisSchema,
        generate_content_config=DEFAULT_GEN_CONFIG,
    )


def create_foucault_analyst() -> Agent:
    return Agent(
        name="FoucaultAnalyst",
        model=get_model(),
        instruction=FOUCAULT_ANALYST_INSTRUCTION,
        tools=[mcp_toolset],
        output_key="foucault_analysis",
        output_schema=FoucaultAnalysisSchema,
        generate_content_config=DEFAULT_GEN_CONFIG,
    )


def create_grounding_evaluator() -> Agent:
    return Agent(
        name="GroundingEvaluator",
        model=get_model(),
        instruction=GROUNDING_EVALUATOR_INSTRUCTION,
        output_key="grounding_evaluation",
        output_schema=GroundingEvaluationSchema,
        generate_content_config=DEFAULT_GEN_CONFIG,
    )


def create_synthesizer() -> Agent:
    return Agent(
        name="Synthesizer",
        model=get_model(),
        instruction=SYNTHESIZER_INSTRUCTION,
        output_key="final_report",
        output_schema=SynthesisReport,
        generate_content_config=SYNTHESIZER_GEN_CONFIG,
    )

# Define individual agent instances for the workflow
input_agent = create_input_agent()
security_auditor = create_security_auditor()
pareto_analyst = create_pareto_analyst()
sowell_analyst = create_sowell_analyst()
mass_psych_analyst = create_mass_psych_analyst()
foucault_analyst = create_foucault_analyst()
grounding_evaluator = create_grounding_evaluator()
synthesizer = create_synthesizer()

# Define the classification node for early-stopping
@node(name="classify_input")
async def classify_input_node(ctx: Context, article_text: str, security_evaluation: dict) -> str:
    # 1. Security Check
    is_safe = security_evaluation.get("is_safe", True)
    # 2. Out of Scope/Satire Check
    if not is_safe or "[OUT_OF_SCOPE]" in article_text or "[SATIRE]" in article_text:
        ctx.route = "skip_analysts"
    else:
        ctx.route = "in_scope"
    return article_text

# Define the grounding classification node
@node(name="classify_grounding")
async def classify_grounding_node(ctx: Context, grounding_evaluation: dict) -> dict:
    is_grounded = grounding_evaluation.get("is_grounded", True)
    # Always route to synthesis path; state grounding_evaluation will differentiate
    ctx.route = "synthesis"
    return grounding_evaluation

# Define workflow graph edges
edges = [
    (START, input_agent),
    (input_agent, security_auditor),
    (security_auditor, classify_input_node),
    (classify_input_node, {
        "in_scope": pareto_analyst,
        "skip_analysts": synthesizer
    }),
    (pareto_analyst, sowell_analyst),
    (sowell_analyst, mass_psych_analyst),
    (mass_psych_analyst, foucault_analyst),
    (foucault_analyst, grounding_evaluator),
    (grounding_evaluator, classify_grounding_node),
    (classify_grounding_node, {
        "synthesis": synthesizer
    })
]

# Create the graph-based Workflow
root_agent = Workflow(
    name="political_analysis_pipeline",
    edges=edges,
)

# Export the app
app = App(
    root_agent=root_agent,
    name="app",
)
