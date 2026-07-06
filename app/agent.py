import os
from pydantic import BaseModel

from dotenv import load_dotenv
from google.adk.agents import Agent, SequentialAgent
from google.adk.apps import App
from google.adk.models import Gemini
from google.genai import types

from app.mcp_client import get_mcp_toolset
from app.tools import fetch_web_page
from app.prompts import (
    INPUT_AGENT_INSTRUCTION,
    PARETO_ANALYST_INSTRUCTION,
    SOWELL_ANALYST_INSTRUCTION,
    MASS_PSYCH_ANALYST_INSTRUCTION,
    FOUCAULT_ANALYST_INSTRUCTION,
    SYNTHESIZER_INSTRUCTION,
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


def create_synthesizer() -> Agent:
    return Agent(
        name="Synthesizer",
        model=get_model(),
        instruction=SYNTHESIZER_INSTRUCTION,
        output_key="final_report",
        output_schema=SynthesisReport,
        generate_content_config=SYNTHESIZER_GEN_CONFIG,
    )


# Wire the sub-agents together sequentially to prevent API free-tier rate limits
root_agent = SequentialAgent(
    name="political_analysis_pipeline",
    sub_agents=[
        create_input_agent(),
        create_pareto_analyst(),
        create_sowell_analyst(),
        create_mass_psych_analyst(),
        create_foucault_analyst(),
        create_synthesizer(),
    ],
)

# Export the app
app = App(
    root_agent=root_agent,
    name="app",
)
