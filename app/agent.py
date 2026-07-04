import os

from dotenv import load_dotenv
from google.adk.agents import Agent, SequentialAgent
from google.adk.apps import App
from google.adk.models import Gemini
from google.genai import types

from app.mcp_client import get_mcp_toolset
from app.tools import fetch_web_page

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


def get_model():
    return Gemini(model=MODEL_NAME, retry_options=types.HttpRetryOptions(attempts=3))


# Callback to prevent rate limits on the AI Studio free tier (5 RPM)
async def rate_limit_delay(callback_context) -> None:
    """No delay needed for gemini-1.5-flash free tier (15 RPM)."""
    pass


def create_input_agent() -> Agent:
    return Agent(
        name="InputAgent",
        model=get_model(),
        instruction=(
            "You are an input processor. The user has provided an input which may be a URL or a text. "
            "If the input looks like a URL (starts with http://, https://, or www.), use the fetch_web_page tool to get the page content, "
            "and output the retrieved clean text of the webpage. If it is already a text, output it. "
            "Do not summarize or analyze it yet; just output the raw text for the other agents to process."
        ),
        tools=[fetch_web_page],
        output_key="article_text",
    )


def create_pareto_analyst() -> Agent:
    return Agent(
        name="ParetoAnalyst",
        model=get_model(),
        before_agent_callback=rate_limit_delay,
        instruction=(
            "You are an expert sociologist specializing in Vilfredo Pareto's theories. Analyze the political text stored in {article_text}. "
            "First, call the get_framework_definition tool for 'pareto' to retrieve the grounding concepts. "
            "CRITICAL DIRECTIVE: Your analysis MUST focus on the CIRCULATION OF ELITES. Do not just lazily map individuals to Foxes or Lions. "
            "Identify the decaying ruling elite and the rising counter-elite. How are they weaponizing residues (Class I: Instinct for Combinations vs Class II: Group Persistence) to maintain or seize power? "
            "Identify the derivations (rhetorical rationalizations) used to mask these structural power grabs. "
            "Provide specific quotes from the text as evidence and explain how they map to Pareto's concepts. "
            "Only perform the analysis. Do NOT call save_analysis_report and do NOT ask if you should save the report."
        ),
        tools=[mcp_toolset],
        output_key="pareto_analysis",
    )


def create_sowell_analyst() -> Agent:
    return Agent(
        name="SowellAnalyst",
        model=get_model(),
        before_agent_callback=rate_limit_delay,
        instruction=(
            "You are an expert political theorist specializing in Thomas Sowell and Carl Schmitt. Analyze the political text stored in {article_text}. "
            "First, call the get_framework_definition tool for 'sowell' to retrieve the grounding concepts. "
            "Identify whether the discourse reflects the Constrained (Tragic) or Unconstrained (Utopian) Vision. "
            "CRITICAL DIRECTIVE 1 (Sowell): You must analyze the SECOND-ORDER EFFECTS and MATERIAL CONSTRAINTS. Do not just label the vision. Explain how the proposed policies alter incentive structures and what the systemic trade-offs are. "
            "CRITICAL DIRECTIVE 2 (Schmitt): Apply Carl Schmitt's Friend/Enemy distinction. How does this text construct an existential 'Enemy'? Is the conflict treated as a mere debate, or an existential struggle where the enemy must be defeated or excluded? "
            "Cite specific quotes as evidence and explain your reasoning. "
            "Only perform the analysis. Do NOT call save_analysis_report and do NOT ask if you should save the report."
        ),
        tools=[mcp_toolset],
        output_key="sowell_analysis",
    )


def create_mass_psych_analyst() -> Agent:
    return Agent(
        name="MassPsychAnalyst",
        model=get_model(),
        before_agent_callback=rate_limit_delay,
        instruction=(
            "You are an expert in crowd psychology and mass movements, specializing in Gustave Le Bon, Eric Hoffer, and René Girard. Analyze the political text stored in {article_text}. "
            "First, call the get_framework_definition tool for 'le_bon' and 'hoffer' using the get_framework_definition tool to retrieve the grounding concepts. "
            "Identify Le Bonian crowd mechanics (simplification, affirmation, contagion) and Hoffer's True Believer dynamics (frustration, self-renunciation). "
            "CRITICAL DIRECTIVE (Girard): Integrate René Girard's Mimetic Theory. Identify the SCAPEGOAT mechanism. Who is being constructed as the 'Unifying Devil' or scapegoat to purify the community or unify the movement? How is mimetic rivalry driving the conflict? "
            "Cite specific quotes as evidence and explain their psychological impact. "
            "Only perform the analysis. Do NOT call save_analysis_report and do NOT ask if you should save the report."
        ),
        tools=[mcp_toolset],
        output_key="mass_psych_analysis",
    )


def create_foucault_analyst() -> Agent:
    return Agent(
        name="FoucaultAnalyst",
        model=get_model(),
        before_agent_callback=rate_limit_delay,
        instruction=(
            "You are an expert philosopher specializing in Michel Foucault. Analyze the political text stored in {article_text}. "
            "First, call the get_framework_definition tool for 'foucault' to retrieve the grounding concepts. "
            "Analyze the Power/Knowledge dynamics and Regimes of Truth. How does the text pathologize opponents to define the boundaries of 'acceptable' discourse? "
            "STRICT NEGATIVE CONSTRAINT: DO NOT conflate welfare economics (e.g., free buses, child care, housing) with Biopower. Biopower strictly applies to the state's administration of biological life (birth rates, mortality, public health, sanitation). If the text discusses welfare, analyze it through Disciplinary Normalization or governmentality instead. "
            "Focus heavily on how the text acts as a disciplinary mechanism to separate the 'normal' from the 'abnormal' political subject. "
            "Cite specific quotes as evidence. "
            "Only perform the analysis. Do NOT call save_analysis_report and do NOT ask if you should save the report."
        ),
        tools=[mcp_toolset],
        output_key="foucault_analysis",
    )


def create_synthesizer() -> Agent:
    return Agent(
        name="Synthesizer",
        model=get_model(),
        before_agent_callback=rate_limit_delay,
        instruction=(
            "You are a master political analyst and synthesizer. Review the original user input, the source, and the individual analyses:\n"
            "- Original Text: {article_text}\n"
            "- Pareto Analysis: {pareto_analysis}\n"
            "- Sowell Analysis: {sowell_analysis}\n"
            "- Mass Psychology Analysis: {mass_psych_analysis}\n"
            "- Foucault Analysis: {foucault_analysis}\n\n"
            "Combine these into a single, cohesive, premium Markdown report. "
            "The report must have a clear title, subtitle, a short 2-sentence executive summary, and detailed sections for each framework. "
            "CRITICAL FORMATTING RULES FOR OBS RECORDING:\n"
            "1. Use H1 (#) for the main title, H2 (##) for framework sections.\n"
            "2. Use bolding for key theoretical terms (e.g., **Residues**, **Biopower**, **Scapegoat**).\n"
            "3. Use blockquotes (>) for specific text citations.\n"
            "4. Ensure the tone is analytically neutral, rigorous, and devoid of conversational filler.\n"
            "At the end, call the save_analysis_report tool to save the report to the SQLite database. "
            "You MUST pass the title, source, report_md, the short summary, and ALSO pass original_text (from {article_text}), pareto_analysis, sowell_analysis, mass_psych_analysis, and foucault_analysis exactly as provided in the context templates. "
            "Your output must be the final markdown report itself."
        ),
        tools=[mcp_toolset],
        output_key="final_report",
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
