import os
from pydantic import BaseModel

from dotenv import load_dotenv
from google.adk.agents import Agent, SequentialAgent
from google.adk.apps import App
from google.adk.models import Gemini
from google.genai import types

from app.mcp_client import get_mcp_toolset
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
        instruction=(
            "You are an input processor. The user has provided an input which may be a URL or a text. "
            "If the input looks like a URL (starts with http://, https://, or www.), use the fetch_web_page tool to get the page content, "
            "and output the retrieved clean text of the webpage. If it is already a text, output it. "
            "Do not summarize or analyze it yet; just output the raw text for the other agents to process."
        ),
        tools=[fetch_web_page],
        output_key="article_text",
        generate_content_config=DEFAULT_GEN_CONFIG,
    )


def create_pareto_analyst() -> Agent:
    return Agent(
        name="ParetoAnalyst",
        model=get_model(),
        instruction=(
            "You are an expert sociologist specializing in Vilfredo Pareto's theories. Analyze the political text stored in {article_text}. "
            "First, call the get_framework_definition tool for 'pareto' to retrieve the grounding concepts. "
            "CRITICAL DIRECTIVE: Your analysis MUST focus on the CIRCULATION OF ELITES. Do not just lazily map individuals to Foxes or Lions. "
            "Identify the decaying ruling elite and the rising counter-elite. How are they weaponizing residues (Class I: Instinct for Combinations vs Class II: Group Persistence) to maintain or seize power? "
            "Identify the derivations (rhetorical rationalizations) used to mask these structural power grabs. "
            "Provide specific quotes from the text as evidence and explain how they map to Pareto's concepts. "
            "Your output must be a JSON object conforming to the output schema, containing:\n"
            "- analysis_md: The detailed analysis in Markdown format.\n"
            "- fox_drive_score: Integer (0 to 100) indicating Class I Fox Drive vs Class II Lion Drive. Calibrate the score strictly using this rubric:\n"
            "  * 0 to 20: Strict Class II Lion dominance (reliance on force, tradition, conservation, group persistence).\n"
            "  * 40 to 60: Balanced or ambiguous mix of Fox and Lion strategies.\n"
            "  * 80 to 100: Strict Class I Fox dominance (reliance on cunning, combinations, innovation, political persuasion).\n"
            "- derivations_detected: List of derivations detected (from assertion, authority, sentiment, sophistry).\n"
            "Only perform the analysis. Do NOT call save_analysis_report and do NOT ask if you should save the report."
        ),
        tools=[mcp_toolset],
        output_key="pareto_analysis",
        output_schema=ParetoAnalysisSchema,
        generate_content_config=DEFAULT_GEN_CONFIG,
    )


def create_sowell_analyst() -> Agent:
    return Agent(
        name="SowellAnalyst",
        model=get_model(),
        instruction=(
            "You are an expert political theorist specializing in Thomas Sowell and Carl Schmitt. Analyze the political text stored in {article_text}. "
            "First, call the get_framework_definition tool for 'sowell' to retrieve the grounding concepts. "
            "Identify whether the discourse reflects the Constrained (Tragic) or Unconstrained (Utopian) Vision. "
            "CRITICAL DIRECTIVE 1 (Sowell): You must analyze the SECOND-ORDER EFFECTS and MATERIAL CONSTRAINTS. Do not just label the vision. Explain how the proposed policies alter incentive structures and what the systemic trade-offs are. "
            "CRITICAL DIRECTIVE 2 (Schmitt): Apply Carl Schmitt's Friend/Enemy distinction. How does this text construct an existential 'Enemy'? Is the conflict treated as a mere debate, or an existential struggle where the enemy must be defeated or excluded? "
            "Cite specific quotes as evidence and explain your reasoning. "
            "Your output must be a JSON object conforming to the output schema, containing:\n"
            "- analysis_md: The detailed analysis in Markdown format.\n"
            "- unconstrained_score: Integer (0 to 100) indicating Utopian/Unconstrained vision drive. Calibrate the score strictly using this rubric:\n"
            "  * 0 to 20: Pure Constrained (Tragic) vision (explicit focus on trade-offs, systemic processes, limits of human knowledge, rule of law, and institutional tradition).\n"
            "  * 40 to 60: Balanced or ambiguous synthesis of trade-offs and political solutions.\n"
            "  * 80 to 100: Pure Unconstrained (Utopian) vision (focus on direct solutions, expert planning, perfectibility of human nature, and social engineering).\n"
            "- schmitt_intensity: Integer (0 to 100) indicating existential conflict intensity. Calibrate the score strictly using this rubric:\n"
            "  * 0 to 20: Liberal parliamentary debate (conflict seen as mock argument or commercial competition; compromises are readily possible).\n"
            "  * 40 to 60: Highly adversarial, but stops short of absolute existential struggle.\n"
            "  * 80 to 100: Absolute existential conflict (politics framed as a friend-versus-enemy combat; opponents are constructed as an existential threat to be excluded or eliminated).\n"
            "Only perform the analysis. Do NOT call save_analysis_report and do NOT ask if you should save the report."
        ),
        tools=[mcp_toolset],
        output_key="sowell_analysis",
        output_schema=SowellAnalysisSchema,
        generate_content_config=DEFAULT_GEN_CONFIG,
    )


def create_mass_psych_analyst() -> Agent:
    return Agent(
        name="MassPsychAnalyst",
        model=get_model(),
        instruction=(
            "You are an expert in crowd psychology and mass movements, specializing in Gustave Le Bon, Eric Hoffer, and René Girard. Analyze the political text stored in {article_text}. "
            "First, call the get_framework_definition tool for 'le_bon' and 'hoffer' using the get_framework_definition tool to retrieve the grounding concepts. "
            "Identify Le Bonian crowd mechanics (simplification, affirmation, contagion) and Hoffer's True Believer dynamics (frustration, self-renunciation). "
            "CRITICAL DIRECTIVE (Girard): Integrate René Girard's Mimetic Theory. Identify the SCAPEGOAT mechanism. Who is being constructed as the 'Unifying Devil' or scapegoat to purify the community or unify the movement? How is mimetic rivalry driving the conflict? "
            "Cite specific quotes as evidence and explain their psychological impact. "
            "Your output must be a JSON object conforming to the output schema, containing:\n"
            "- analysis_md: The detailed analysis in Markdown format.\n"
            "- mimetic_tension: Integer (0 to 100) indicating mimetic rivalry tension. Calibrate the score strictly using this rubric:\n"
            "  * 0 to 20: Independent desires (cooperation, absence of imitation, diverse/non-competitive political goals).\n"
            "  * 40 to 60: Moderate mimetic imitation or comparison-driven status anxiety.\n"
            "  * 80 to 100: Extreme mimetic crisis (obsessive imitation and fierce, direct competition for identical status/power leading to pure conflict).\n"
            "- scapegoat_index: Integer (0 to 100) indicating scapegoating intensity. Calibrate the score strictly using this rubric:\n"
            "  * 0 to 20: No target singled out (blame is systemic or absent; conflicts are resolved through structured channels).\n"
            "  * 40 to 60: Rhetorical finger-pointing without collective purification rituals.\n"
            "  * 80 to 100: Intense scapegoating (a single person or group is framed as the 'unifying devil' or absolute source of evil whose exclusion is required to bring harmony/unity).\n"
            "Only perform the analysis. Do NOT call save_analysis_report and do NOT ask if you should save the report."
        ),
        tools=[mcp_toolset],
        output_key="mass_psych_analysis",
        output_schema=MassPsychAnalysisSchema,
        generate_content_config=DEFAULT_GEN_CONFIG,
    )


def create_foucault_analyst() -> Agent:
    return Agent(
        name="FoucaultAnalyst",
        model=get_model(),
        instruction=(
            "You are an expert philosopher specializing in Michel Foucault. Analyze the political text stored in {article_text}. "
            "First, call the get_framework_definition tool for 'foucault' to retrieve the grounding concepts. "
            "Analyze the Power/Knowledge dynamics and Regimes of Truth. How does the text pathologize opponents to define the boundaries of 'acceptable' discourse? "
            "STRICT NEGATIVE CONSTRAINT: DO NOT conflate welfare economics (e.g., free buses, child care, housing) with Biopower. Biopower strictly applies to the state's administration of biological life (birth rates, mortality, public health, sanitation). If the text discusses welfare, analyze it through Disciplinary Normalization or governmentality instead. "
            "Focus heavily on how the text acts as a disciplinary mechanism to separate the 'normal' from the 'abnormal' political subject. "
            "Cite specific quotes as evidence. "
            "Your output must be a JSON object conforming to the output schema, containing:\n"
            "- analysis_md: The detailed analysis in Markdown format.\n"
            "- mechanisms_detected: List of mechanisms detected. You MUST restrict your selection strictly to these exact string values: "
            "['truthRegime', 'normalization', 'biopower', 'governmentality', 'pathologization']. Any others will break the dashboard rendering.\n"
            "Only perform the analysis. Do NOT call save_analysis_report and do NOT ask if you should save the report."
        ),
        tools=[mcp_toolset],
        output_key="foucault_analysis",
        output_schema=FoucaultAnalysisSchema,
        generate_content_config=DEFAULT_GEN_CONFIG,
    )


def create_synthesizer() -> Agent:
    return Agent(
        name="Synthesizer",
        model=get_model(),
        instruction=(
            "You are a master political analyst and synthesizer. Review the original user input, the source, and the individual analyses:\n"
            "- Original Text: {article_text}\n"
            "- Pareto Analysis: {pareto_analysis}\n"
            "- Sowell Analysis: {sowell_analysis}\n"
            "- Mass Psychology Analysis: {mass_psych_analysis}\n"
            "- Foucault Analysis: {foucault_analysis}\n\n"
            "Combine these into a single, cohesive, premium JSON report conforming to the requested schema. "
            "The JSON report must have the following fields:\n"
            "- title: A clear, academic-grade title for the analysis.\n"
            "- subtitle: A descriptive subtitle reflecting the main theme.\n"
            "- summary: A short 2-sentence executive summary of the key findings.\n"
            "- report_md: The detailed synthesized report in Markdown format containing sections for each framework.\n\n"
            "CRITICAL FORMATTING RULES FOR report_md:\n"
            "1. Use H2 (##) for framework sections.\n"
            "2. Use bolding for key theoretical terms (e.g., **Residues**, **Biopower**, **Scapegoat**).\n"
            "3. Use blockquotes (>) for specific text citations.\n"
            "4. Ensure the tone is analytically neutral, rigorous, and devoid of conversational filler."
        ),
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
