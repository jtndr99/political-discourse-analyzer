import asyncio
import json
import os
import sqlite3

import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server

# Initialize SQLite database
DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "analyses.db"
)


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            source TEXT NOT NULL,
            report_md TEXT NOT NULL,
            summary TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Check and add new columns if they do not exist (migration logic)
    cursor.execute("PRAGMA table_info(analyses)")
    columns = [col[1] for col in cursor.fetchall()]

    new_cols = {
        "original_text": "TEXT",
        "pareto_analysis": "TEXT",
        "sowell_analysis": "TEXT",
        "mass_psych_analysis": "TEXT",
        "foucault_analysis": "TEXT",
    }

    for col_name, col_type in new_cols.items():
        if col_name not in columns:
            cursor.execute(f"ALTER TABLE analyses ADD COLUMN {col_name} {col_type}")

    conn.commit()
    conn.close()


# Framework reference library grounding text
FRAMEWORKS = {
    "pareto": """Vilfredo Pareto's Theory of Residues and Derivations:
- Residues: The underlying, persistent, non-logical sentiments, drives, or instincts that motivate human behavior.
  * Class I: Instinct for Combinations: The drive to synthesize, innovate, associate different things, use cunning, and build complex systems (often associated with 'Foxes' who rule by intellect and persuasion).
  * Class II: Group Persistence: The conservative drive to maintain social structures, family, nation, tradition, and shared identity (associated with 'Lions' who rule by force, order, and loyalty).
- Derivations: The intellectual justifications, rationalizations, or explanations used to make non-logical actions appear logical.
  * Class I: Assertion: Simple, dogmatic statements presented as self-evident truths (e.g., 'This is the only way').
  * Class II: Authority: Appealing to prestige, status, respected figures, or sacred texts (e.g., 'As the Constitution states...').
  * Class III: Accords with Sentiments or Principles: Appealing to shared moral values, justice, human rights, or national interest (e.g., 'For the sake of fairness...').
  * Class IV: Verbal Proofs: Wordplay, sophistry, vague concepts, and logical fallacies designed to persuade without rigorous proof (e.g., using terms like 'solidarity' or 'common sense' to obscure details).""",
    "sowell": """Thomas Sowell's Theory of Political Visions:
- The Constrained Vision (Tragic Vision):
  * Human Nature: Human nature is inherently limited, flawed, and fixed. Self-interest and moral limitations are permanent constraints of human life.
  * Solution: Instead of trying to change human nature, society must rely on evolved institutions, trade-offs, and systemic rules (e.g., markets, rule of law, tradition) to channel self-interest toward social order.
  * Key Focus: Systemic processes, individual freedom within rules, trade-offs rather than absolute solutions, suspicion of concentrated power.
- The Unconstrained Vision (Utopian Vision):
  * Human Nature: Human nature is malleable, perfectible, and capable of direct altruism. Humans can be improved through education and social engineering.
  * Solution: Social problems are caused by bad institutions and can be directly solved through reason, planning, and the leadership of wise, public-spirited individuals.
  * Key Focus: Intentions, direct action, absolute solutions ('social justice'), social engineering, confidence in expert decision-making.""",
    "le_bon": """Gustave Le Bon's Crowd Psychology:
- Crowd Mind: When individuals assemble, their conscious personalities dissolve, and a collective mind forms, driven by the unconscious. Nuance is lost, and the crowd acts on raw, collective impulse.
- Suggestibility and Contagion: Ideas, feelings, and beliefs spread rapidly through emotional contagion and suggestibility. The crowd is highly open to influence, reacting to images and slogans rather than logic.
- Simplification: Crowds think in absolute, black-and-white terms. Ideas must be simplified to their most extreme forms to be accepted. There is no room for doubt or critical thinking.
- Affirmation, Repetition, and Contagion: Leaders influence crowds not through reasoning, but through:
  * Affirmation: Making absolute, positive assertions free of reasoning or proof.
  * Repetition: Repeating the same assertion constantly until it enters the unconscious mind as an undisputed truth.""",
    "hoffer": """Eric Hoffer's Mass Movement Analysis:
- The True Believer: The fanatical follower who seeks to escape a frustrated, meaningless, or failed personal existence by fully merging with a mass movement and its holy cause.
- Frustration and Self-Renunciation: True believers do not want self-improvement; they want self-renunciation. They seek a new identity, absolute unity, and a license to act without individual conscience.
- The Unifying Devil (The Common Enemy): A shared hatred of a designated enemy is a powerful unifying agent. Hoffer noted that mass movements can spread without a belief in a God, but never without belief in a devil.
- Doctrine vs. Facts: The doctrine of a mass movement is accepted as absolute truth. The truer the belief, the less it relies on facts or logical coherence. Indeed, the readiness to believe the absurd is a test of devotion.""",
    "foucault": """Michel Foucault's Power/Knowledge and Discourse Dynamics:
- Power/Knowledge (Pouvoir/Savoir): Power and knowledge are co-constitutive. Knowledge is not neutral or disinterested; it is produced within power relations and is used to reinforce those relations. Conversely, power is exercised through the production of knowledge (e.g., defining what is 'rational', 'normal', or 'scientific').
- Regimes of Truth: Every society has a regime of truth—rules and mechanisms that determine which discourses are accepted and function as true, how statements are validated, and who is authorized to speak with truth.
- Biopower / Biopolitics: Power that regulates, manages, and subjugates human life at the level of populations (e.g., birth rates, public health, demographic control, public hygiene) as a state resource.
- Disciplinary Power & Normalization: Power that operates at the individual level through surveillance, classification, examination, and the enforcement of norms, separating the 'normal' from the 'abnormal' (clinical, educational, penal systems).""",
}

# Create MCP Server instance
server = Server("discourse-anal-mcp")


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="get_framework_definition",
            description="Retrieve detailed grounding and definition texts for a classical sociological/philosophical framework (pareto, sowell, le_bon, hoffer, foucault). Use this to ground the analysis in accurate definitions.",
            inputSchema={
                "type": "object",
                "properties": {
                    "framework": {
                        "type": "string",
                        "enum": ["pareto", "sowell", "le_bon", "hoffer", "foucault"],
                        "description": "The sociological/philosophical framework name.",
                    }
                },
                "required": ["framework"],
            },
        ),
        types.Tool(
            name="save_analysis_report",
            description="Save a completed political discourse analysis report to the local SQLite database.",
            inputSchema={
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Title of the analyzed article or speech.",
                    },
                    "source": {
                        "type": "string",
                        "description": "Source URL or paste description.",
                    },
                    "report_md": {
                        "type": "string",
                        "description": "The complete synthesized analysis report in Markdown format.",
                    },
                    "summary": {
                        "type": "string",
                        "description": "A short 1-2 sentence summary of the key findings.",
                    },
                    "original_text": {
                        "type": "string",
                        "description": "The full original text/article analyzed.",
                    },
                    "pareto_analysis": {
                        "type": "string",
                        "description": "Pareto specialist analysis report.",
                    },
                    "sowell_analysis": {
                        "type": "string",
                        "description": "Sowell specialist analysis report.",
                    },
                    "mass_psych_analysis": {
                        "type": "string",
                        "description": "Crowd psychology specialist analysis report.",
                    },
                    "foucault_analysis": {
                        "type": "string",
                        "description": "Foucault specialist analysis report.",
                    },
                },
                "required": ["title", "source", "report_md"],
            },
        ),
        types.Tool(
            name="list_analysis_reports",
            description="Retrieve a list of all saved discourse analysis reports from the SQLite database.",
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="get_analysis_report",
            description="Retrieve the full markdown report and details of a specific saved analysis.",
            inputSchema={
                "type": "object",
                "properties": {
                    "report_id": {
                        "type": "integer",
                        "description": "The ID of the saved report.",
                    }
                },
                "required": ["report_id"],
            },
        ),
    ]


@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent]:
    arguments = arguments or {}

    if name == "get_framework_definition":
        framework = arguments.get("framework")
        if framework not in FRAMEWORKS:
            return [
                types.TextContent(
                    type="text",
                    text=f"Error: Unknown framework '{framework}'. Valid frameworks: {list(FRAMEWORKS.keys())}",
                )
            ]
        return [types.TextContent(type="text", text=FRAMEWORKS[framework])]

    elif name == "save_analysis_report":
        title = arguments.get("title")
        source = arguments.get("source")
        report_md = arguments.get("report_md")
        summary = arguments.get("summary", "")
        original_text = arguments.get("original_text", "")
        pareto_analysis = arguments.get("pareto_analysis", "")
        sowell_analysis = arguments.get("sowell_analysis", "")
        mass_psych_analysis = arguments.get("mass_psych_analysis", "")
        foucault_analysis = arguments.get("foucault_analysis", "")

        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO analyses (title, source, report_md, summary, original_text, pareto_analysis, sowell_analysis, mass_psych_analysis, foucault_analysis) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    title,
                    source,
                    report_md,
                    summary,
                    original_text,
                    pareto_analysis,
                    sowell_analysis,
                    mass_psych_analysis,
                    foucault_analysis,
                ),
            )
            report_id = cursor.lastrowid
            conn.commit()
            conn.close()
            return [
                types.TextContent(
                    type="text",
                    text=f"Successfully saved analysis report. ID: {report_id}",
                )
            ]
        except Exception as e:
            return [
                types.TextContent(
                    type="text", text=f"Error saving report to database: {e!s}"
                )
            ]

    elif name == "list_analysis_reports":
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

            return [types.TextContent(type="text", text=json.dumps(reports, indent=2))]
        except Exception as e:
            return [
                types.TextContent(type="text", text=f"Error listing reports: {e!s}")
            ]

    elif name == "get_analysis_report":
        report_id = arguments.get("report_id")
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
                return [
                    types.TextContent(
                        type="text",
                        text=f"Error: Report with ID {report_id} not found.",
                    )
                ]

            report = {
                "id": row[0],
                "title": row[1],
                "source": row[2],
                "report_md": row[3],
                "summary": row[4],
                "created_at": row[5],
                "original_text": row[6],
                "pareto_analysis": row[7],
                "sowell_analysis": row[8],
                "mass_psych_analysis": row[9],
                "foucault_analysis": row[10],
            }
            return [types.TextContent(type="text", text=json.dumps(report, indent=2))]
        except Exception as e:
            return [
                types.TextContent(type="text", text=f"Error retrieving report: {e!s}")
            ]

    raise ValueError(f"Unknown tool: {name}")


async def main():
    # Initialize DB schema
    init_db()

    # Run the server using stdio transport
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream, server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
