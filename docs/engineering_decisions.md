# Engineering Decisions

This document outlines the core engineering decisions made during the design, development, and scaling of the **Political Discourse Analyzer** multi-agent application.

---

## 1. Orchestration Framework: Google Agent Development Kit (ADK)
### Decision
We selected the **Google Agent Development Kit (ADK)** as the primary multi-agent orchestration framework over alternatives like LangChain, Autogen, or CrewAI.

### Rationale
* **Native Gemini Integration**: The ADK provides first-class support for the Google GenAI SDK, which guarantees optimal token utilization, precise response schemas, and native support for low-latency calls to Gemini models.
* **Low Orchestration Overhead**: Heavyweight frameworks like CrewAI introduce complex graph runners and high abstraction layers. The ADK's `SequentialAgent` pipeline allows us to propagate state from agent to agent cleanly using session variables (e.g. `{article_text}`, `{pareto_analysis}`).
* **State Management**: ADK's built-in `InMemorySessionService` acts as a thread-safe state ledger, simplifying the aggregation of intermediate specialist reports.

---

## 2. Execution Design: Sequential Orchestration
### Decision
Specialist agents execute **sequentially** rather than concurrently.

### Rationale
* **Rate Limit Mitigation**: The Gemini free tier has a limit of 15 Requests Per Minute (RPM). Running four specialist analysts and a synthesizer in parallel would trigger simultaneous API requests, causing immediate rate limit exhaustion. Sequential execution distributes calls evenly over time.
* **Proactive Protection**: Combined with our global token-bucket `RateLimiterPlugin` in `app/__init__.py`, sequential execution guarantees that requests pace themselves naturally and pause gracefully when rate thresholds are approached, instead of crashing mid-run.
* **Coherent State Growth**: Each specialist analyst builds upon the sanitized output of the `InputAgent` stored in the session ledger, ensuring that downstream synthesizers can reliably query structural states without race conditions.

---

## 3. Real-Time Streaming: Server-Sent Events (SSE)
### Decision
We implemented unidirectional real-time progress streaming using **Server-Sent Events (SSE)** via FastAPI's native `StreamingResponse`.

### Rationale
* **SSE vs WebSockets**: WebSockets are bi-directional and introduce high connection handshake overhead, stateful management difficulties, and firewall traversal complications. Because the client only needs to receive progress notifications and stream data unidirectionally from the server, SSE is the correct architectural pattern.
* **SSE vs HTTP Polling**: Polling the database from the client introduces high latency, database read overhead, and poor user experience. SSE streams progress chunks to the client immediately as soon as a specialist agent finishes writing its session state.
* **Native Implementation**: Using FastAPI's native generator-based `StreamingResponse` allowed us to stream standard compliant `data: ...` blocks without adding third-party dependencies like `sse-starlette` to the environment.

---

## 4. Grounding Knowledge: Model Context Protocol (MCP)
### Decision
We built a custom local **Model Context Protocol (MCP)** server (`app/mcp_server.py`) to supply framework definitions to the specialist agents.

### Rationale
* **Context Alignment**: Different LLMs hold varied understandings of sociological concepts. Grounding the agents with concrete definitions of Pareto's *Residues/Derivations*, Sowell's *Political Visions*, and Girard's *Scapegoating* ensures high conceptual accuracy.
* **Decoupling Logic**: Moving definition texts and reference articles out of agent prompts and into a dedicated database/lookup service allows us to update definitions or expand reference materials without touching core agent source code.

---

## 5. Storage and Saving: Programmatic FastAPI SQLite Writes
### Decision
We write the final analysis report to the SQLite database programmatically in the FastAPI handler rather than having the `Synthesizer` call an MCP tool.

### Rationale
* **Process Separation**: Keeping agent logic focused entirely on reasoning and report generation, while leaving persistence to the FastAPI server, preserves a clean separation of concerns.
* **FastAPI Stream Synchronization**: To ensure database writes only occur upon *successful* completion of the entire multi-agent loop, saving is executed at the end of the SSE stream generator. This prevents orphaned partial reports from populating the database in the event of mid-run failures.
* **Canonical Persistence**: Prevents database connection deadlocks and schema duplication across multiple execution processes.
