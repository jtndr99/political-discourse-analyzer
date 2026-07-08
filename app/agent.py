import os

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.agents.context import Context
from google.adk.apps import App
from google.adk.models import Gemini
from google.adk.workflow import START, Workflow, node
from google.genai import types
from pydantic import BaseModel

from app.mcp_client import get_mcp_toolset
from app.prompts import (
    FOUCAULT_ANALYST_INSTRUCTION,
    GROUNDING_EVALUATOR_INSTRUCTION,
    MASS_PSYCH_ANALYST_INSTRUCTION,
    PARETO_ANALYST_INSTRUCTION,
    SCOPE_CLASSIFIER_INSTRUCTION,
    SECURITY_AUDITOR_INSTRUCTION,
    SOWELL_ANALYST_INSTRUCTION,
    SYNTHESIZER_INSTRUCTION,
)
from app.tools import fetch_web_page


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


class ScopeClassificationSchema(BaseModel):
    is_out_of_scope: bool
    is_satire: bool
    reasoning: str


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


def create_scope_classifier() -> Agent:
    return Agent(
        name="ScopeClassifier",
        model=get_model(),
        instruction=SCOPE_CLASSIFIER_INSTRUCTION,
        # NOTE: no tools, no freeform text output field — output_schema restricts
        # this agent to booleans + one metadata sentence. It has no channel through
        # which it could emit fabricated "article" content, by construction.
        output_key="scope_classification",
        output_schema=ScopeClassificationSchema,
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
scope_classifier = create_scope_classifier()
security_auditor = create_security_auditor()
pareto_analyst = create_pareto_analyst()
sowell_analyst = create_sowell_analyst()
mass_psych_analyst = create_mass_psych_analyst()
foucault_analyst = create_foucault_analyst()
grounding_evaluator = create_grounding_evaluator()
synthesizer = create_synthesizer()


# Deterministic, zero-LLM capture of the user's raw input. This is the ONLY
# place `raw_input_text` is ever written. No model sees this text before it's
# stored, so nothing downstream can rewrite, launder, or fabricate it.
@node(name="capture_raw_input")
async def capture_raw_input_node(ctx: Context, node_input: str) -> str:
    text = (node_input or "").strip()
    if text.startswith(("http://", "https://")):
        result = fetch_web_page(text)
        if result.get("status") == "success":
            captured = result.get("text", "")
        else:
            captured = (
                f"[FETCH_ERROR] {result.get('message', 'Unknown error fetching URL')}"
            )
    else:
        captured = text
    ctx.actions.state_delta["raw_input_text"] = captured
    return captured


# Deterministic tag application. `article_text` is ALWAYS exactly the captured
# raw text, optionally prefixed with [OUT_OF_SCOPE]/[SATIRE]. No LLM writes
# this value directly — ScopeClassifier only supplies the two booleans that
# decide which prefix (if any) gets applied by plain Python.
@node(name="tag_article")
async def tag_article_node(
    ctx: Context, raw_input_text: str, scope_classification: dict
) -> str:
    prefix = ""
    if scope_classification.get("is_out_of_scope"):
        prefix += "[OUT_OF_SCOPE] "
    if scope_classification.get("is_satire"):
        prefix += "[SATIRE] "
    article_text = prefix + raw_input_text
    ctx.actions.state_delta["article_text"] = article_text
    return article_text


# Define the classification node for early-stopping
@node(name="classify_input")
async def classify_input_node(
    ctx: Context, article_text: str, security_evaluation: dict
) -> str:
    # 1. Security Check
    is_safe = security_evaluation.get("is_safe", True)
    # 2. Out of Scope/Satire Check
    if not is_safe or "[OUT_OF_SCOPE]" in article_text or "[SATIRE]" in article_text:
        ctx.route = "skip_analysts"
    else:
        ctx.route = "in_scope"
    return article_text


# Define workflow graph edges.
# SecurityAuditor now runs on raw_input_text (captured before any LLM touches
# it), so it can no longer be defeated by an upstream agent laundering the
# attack into fabricated "safe-looking" content.
edges = [
    (START, capture_raw_input_node),
    (capture_raw_input_node, security_auditor),
    (security_auditor, scope_classifier),
    (scope_classifier, tag_article_node),
    (tag_article_node, classify_input_node),
    (classify_input_node, {"in_scope": pareto_analyst, "skip_analysts": synthesizer}),
    (pareto_analyst, sowell_analyst),
    (sowell_analyst, mass_psych_analyst),
    (mass_psych_analyst, foucault_analyst),
    (foucault_analyst, grounding_evaluator),
    (grounding_evaluator, synthesizer),
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
