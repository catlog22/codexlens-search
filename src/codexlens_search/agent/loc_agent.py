from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from codexlens_search.agent.tools import get_tool_schemas, get_graph_enhanced_tool_schemas
from codexlens_search.config import Config
from codexlens_search.core.entity import EntityId, EntityKind
from codexlens_search.core.entity_graph import EntityGraph
from codexlens_search.search.pipeline import FileSearchResult, SearchPipeline

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM client abstraction — openai SDK only, supports any OpenAI-compatible API
# (GLM/智谱, DeepSeek, Qwen, vLLM, Ollama, etc.)
# ---------------------------------------------------------------------------

def _create_openai_client(api_key: str, api_base: str) -> Any:
    """Create an OpenAI client. Returns None if openai is not installed or no key."""
    try:
        from openai import OpenAI
    except ImportError:
        _log.warning("openai package not installed — agent LLM disabled")
        return None

    # Need either an explicit key or OPENAI_API_KEY in env
    resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not resolved_key:
        _log.warning("No API key for agent LLM (set agent_llm_api_key or OPENAI_API_KEY)")
        return None

    kwargs: dict[str, Any] = {"api_key": resolved_key}
    if api_base:
        kwargs["base_url"] = api_base
    return OpenAI(**kwargs)


def _create_async_openai_client(api_key: str, api_base: str) -> Any:
    """Create an AsyncOpenAI client. Returns None if openai is not installed or no key."""
    try:
        from openai import AsyncOpenAI
    except ImportError:
        _log.warning("openai package not installed — async agent LLM disabled")
        return None

    resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not resolved_key:
        _log.warning("No API key for agent LLM (set agent_llm_api_key or OPENAI_API_KEY)")
        return None

    kwargs: dict[str, Any] = {"api_key": resolved_key}
    if api_base:
        kwargs["base_url"] = api_base
    return AsyncOpenAI(**kwargs)


async def _call_openai_async(
    client: Any,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Call AsyncOpenAI-compatible chat completion with tool use.

    Returns the assistant message dict or None on failure.
    """
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            timeout=120,
        )
    except Exception:
        _log.warning("Async OpenAI completion failed", exc_info=True)
        return None

    choice = resp.choices[0] if resp.choices else None
    if choice is None:
        return None

    msg = choice.message
    out: dict[str, Any] = {
        "role": "assistant",
        "content": msg.content or "",
    }
    if msg.tool_calls:
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in msg.tool_calls
        ]
    return out


def _call_openai(
    client: Any,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Call OpenAI-compatible chat completion with tool use.

    Returns the assistant message dict or None on failure.
    """
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            timeout=120,
        )
    except Exception:
        _log.warning("OpenAI completion failed", exc_info=True)
        return None

    choice = resp.choices[0] if resp.choices else None
    if choice is None:
        return None

    msg = choice.message
    out: dict[str, Any] = {
        "role": "assistant",
        "content": msg.content or "",
    }
    if msg.tool_calls:
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in msg.tool_calls
        ]
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_path_like(value: str) -> bool:
    return "/" in value or "\\" in value


# Symbol extraction from code text
_SYMBOL_PATTERN = re.compile(
    r"(?:def|class)\s+([a-zA-Z_][a-zA-Z0-9_]+)",
)
_IMPORT_PATTERN = re.compile(
    r"(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))",
)


def _extract_symbols_from_text(text: str) -> list[str]:
    """Extract function/class names and import sources from code text."""
    seen: set[str] = set()
    symbols: list[str] = []
    for match in _SYMBOL_PATTERN.finditer(text):
        name = match.group(1)
        # Skip very common/short names that add noise
        if len(name) > 3 and name not in seen and not name.startswith("_"):
            seen.add(name)
            symbols.append(name)
    return symbols[:20]  # cap to avoid overwhelming the LLM


# Import extraction from code text
_IMPORT_FROM_RE = re.compile(r"^\s*from\s+([\w.]+)\s+import", re.MULTILINE)
_IMPORT_DIRECT_RE = re.compile(r"^\s*import\s+([\w.]+)", re.MULTILINE)


def _extract_imports_from_text(text: str) -> list[str]:
    """Extract dotted module paths from Python import statements."""
    seen: set[str] = set()
    modules: list[str] = []
    for pattern in (_IMPORT_FROM_RE, _IMPORT_DIRECT_RE):
        for match in pattern.finditer(text):
            mod = match.group(1)
            # Skip single-name imports (likely stdlib: os, sys, json, etc.)
            if "." in mod and mod not in seen:
                seen.add(mod)
                modules.append(mod)
    return modules


def _resolve_module_to_path(module: str, cwd: Path) -> str | None:
    """Resolve a dotted Python module to a project-local file path."""
    parts = module.split(".")
    # Direct: chainlit.socket → chainlit/socket.py
    direct = "/".join(parts) + ".py"
    if (cwd / direct).is_file():
        return direct
    # Package __init__: chainlit.socket → chainlit/socket/__init__.py
    pkg_init = "/".join(parts) + "/__init__.py"
    if (cwd / pkg_init).is_file():
        return pkg_init
    # Glob fallback: handles prefix directories (e.g., backend/chainlit/socket.py)
    if len(parts) >= 2:
        suffix = "/".join(parts[-2:]) + ".py"
        for p in cwd.glob(f"**/{suffix}"):
            if ".git" not in p.parts and "__pycache__" not in p.parts:
                return str(p.relative_to(cwd)).replace("\\", "/")
    return None


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


# File path extraction from conversation history
_PATH_PATTERN = re.compile(
    r"""(?:^|[\s"'`\[({,])"""         # boundary before path
    r"""((?:[\w./\\-]+/)?"""           # optional directory prefix with /
    r"""[\w.-]+\."""                    # filename with dot
    r"""(?:py|js|ts|jsx|tsx|go|java|cpp|c|h|hpp|cs|rs|rb|php|scala|kt|swift"""
    r"""|lua|sh|bash|vue|svelte|yaml|yml|json|toml|cfg|ini|md|txt|html|css)"""
    r""")"""                           # extension
    r"""(?:$|[\s"'`\])},:])""",        # boundary after path
    re.MULTILINE,
)


def _extract_file_paths_from_messages(messages: list[dict[str, Any]]) -> list[str]:
    """Extract file paths mentioned in tool results and assistant messages."""
    seen: set[str] = set()
    paths: list[str] = []

    for msg in messages:
        role = msg.get("role", "")
        if role not in ("assistant", "tool"):
            continue

        text = ""
        if role == "assistant":
            text = msg.get("content", "") or ""
        elif role == "tool":
            text = msg.get("content", "") or ""

        # Also check tool call arguments for file paths
        for tc in msg.get("tool_calls", []):
            fn = tc.get("function", {})
            text += " " + (fn.get("arguments", "") or "")

        for match in _PATH_PATTERN.finditer(text):
            p = match.group(1).strip()
            if p and p not in seen:
                seen.add(p)
                paths.append(p)

    return paths


def _extract_paths_from_search_results(messages: list[dict[str, Any]]) -> list[str]:
    """Extract file paths from search_code tool results (JSON arrays)."""
    seen: set[str] = set()
    paths: list[str] = []

    for msg in messages:
        if msg.get("role") != "tool":
            continue
        content = msg.get("content", "") or ""
        if not content.startswith("["):
            continue
        try:
            results = json.loads(content)
        except Exception:
            continue
        if not isinstance(results, list):
            continue
        for item in results:
            if isinstance(item, dict):
                p = item.get("path", "")
                if p and p not in seen:
                    seen.add(p)
                    paths.append(p)

    return paths


# ---------------------------------------------------------------------------
# CodeLocAgent
# ---------------------------------------------------------------------------

class CodeLocAgent:
    """LLM-driven iterative code localization agent.

    Uses the openai SDK to call any OpenAI-compatible API (including GLM,
    DeepSeek, Qwen, etc.) for tool-use based code localization.

    Termination: inspired by pi-mono's agent loop — when the LLM responds
    without any tool calls, the loop stops naturally. File paths are extracted
    from the conversation history instead of requiring a special "finish" tool.
    """

    def __init__(
        self,
        search_pipeline: SearchPipeline,
        entity_graph: EntityGraph | None,
        config: Config,
    ) -> None:
        self._search = search_pipeline
        self._entity_graph = entity_graph
        self._config = config
        self._tool_schemas = get_tool_schemas()
        self._client: Any = None  # lazy sync client
        self._async_client: Any = None  # lazy async client
        self._discovered_edges: list[dict[str, Any]] = []

    def _get_client(self) -> Any | None:
        if self._client is None:
            cfg = self._config
            self._client = _create_openai_client(cfg.agent_llm_api_key, cfg.agent_llm_api_base)
        return self._client

    def _get_async_client(self) -> Any | None:
        if self._async_client is None:
            cfg = self._config
            self._async_client = _create_async_openai_client(cfg.agent_llm_api_key, cfg.agent_llm_api_base)
        return self._async_client

    def _should_fan_out(self, query: str) -> bool:
        """Heuristic to determine if a query should be decomposed into sub-queries.

        Returns True for queries with multiple distinct concepts, files, or
        explicit multi-part indicators (AND/OR, comma-separated items).
        Conservative to avoid unnecessary decomposition and LLM cost.
        """
        if not self._config.agent_fan_out_enabled:
            return False

        # Check for explicit multi-part indicators
        lower = query.lower()

        # AND/OR connectors between concepts
        if re.search(r"\b(?:and|or)\b.*\b(?:and|or)\b", lower):
            return True

        # Comma-separated list of 3+ items (not inside parentheses)
        commas = query.count(",")
        if commas >= 2:
            return True

        # Multiple file references (e.g., "foo.py and bar.py")
        file_refs = re.findall(
            r"[\w./\\-]+\.(?:py|js|ts|jsx|tsx|go|java|cpp|c|h|rs|rb)", query
        )
        if len(file_refs) >= 2:
            return True

        return False

    async def _fan_out(
        self, query: str, max_iterations: int, top_k: int,
    ) -> list[FileSearchResult]:
        """Decompose query into sub-queries, run parallel agents, merge results."""
        cfg = self._config
        client = self._get_async_client()
        if client is None:
            return await self._run_single(query, max_iterations, top_k)

        # Ask LLM to decompose the query
        decompose_messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You are a query decomposition assistant. Split the user's code "
                    "localization query into 2-4 independent sub-queries, each focusing "
                    "on a single concept or file. Output ONLY a JSON array of strings. "
                    "Example: [\"find auth login handler\", \"find session middleware\"]"
                ),
            },
            {"role": "user", "content": query},
        ]
        try:
            resp = await client.chat.completions.create(
                model=cfg.agent_llm_model,
                messages=decompose_messages,
                timeout=30,
            )
            raw = (resp.choices[0].message.content or "").strip()
            # Extract JSON array from response
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            if match:
                sub_queries = json.loads(match.group())
            else:
                sub_queries = []
        except Exception:
            _log.warning("Fan-out decomposition failed, falling back to single agent", exc_info=True)
            return await self._run_single(query, max_iterations, top_k)

        if not sub_queries or len(sub_queries) < 2:
            return await self._run_single(query, max_iterations, top_k)

        # Cap sub-queries
        max_workers = cfg.agent_fan_out_max_workers
        sub_queries = sub_queries[:max_workers]

        _log.info("Fan-out: decomposed into %d sub-queries: %s", len(sub_queries), sub_queries)

        # Run sub-agents in parallel
        tasks = [
            self._run_single(sq, max_iterations, top_k)
            for sq in sub_queries
        ]
        all_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Merge and deduplicate by file path
        seen_paths: set[str] = set()
        merged: list[FileSearchResult] = []
        for result in all_results:
            if isinstance(result, Exception):
                _log.warning("Sub-agent failed: %s", result)
                continue
            for r in result:
                if r.path not in seen_paths:
                    seen_paths.add(r.path)
                    merged.append(r)

        # Sort by score descending and cap
        merged.sort(key=lambda x: x.score, reverse=True)
        return merged[:top_k]

    async def run(self, query: str, max_iterations: int = 5, top_k: int = 10) -> list[FileSearchResult]:
        """Run the agent loop asynchronously using AsyncOpenAI.

        Use ``run_sync()`` from synchronous callers (e.g. CLI bridge).
        When fan-out is enabled and the query contains multiple concepts,
        decomposes the query and runs parallel sub-agents.

        Supports three modes (via ``config.agent_mode``):
        - ``"agent"`` (default): original iterative agent loop
        - ``"graph_enhanced"``: pipeline + agent in parallel, agent enriches graph
        - ``"hybrid"``: full agent loop with graph enhancement + pipeline reranking
        """
        max_iterations = max(1, _safe_int(max_iterations, 5))
        top_k = max(1, _safe_int(top_k, 10))

        mode = self._config.agent_mode

        if mode == "graph_enhanced":
            return await self._run_graph_enhanced(query, max_iterations, top_k)

        if mode == "hybrid":
            return await self._run_hybrid(query, max_iterations, top_k)

        # Original agent mode
        if self._should_fan_out(query):
            return await self._fan_out(query, max_iterations, top_k)

        return await self._run_single(query, max_iterations, top_k)

    async def _run_single(self, query: str, max_iterations: int = 5, top_k: int = 10) -> list[FileSearchResult]:
        """Run a single agent loop (no fan-out)."""
        cfg = self._config
        max_iterations = max(1, _safe_int(max_iterations, 5))
        top_k = max(1, _safe_int(top_k, 10))

        if not cfg.agent_enabled:
            _log.debug("Agent disabled, falling back to search_files")
            return self._search.search_files(query, top_k=top_k)

        client = self._get_async_client()
        if client is None:
            _log.warning("No async LLM client available, falling back to search_files")
            return self._search.search_files(query, top_k=top_k)

        system_prompt = self._build_system_prompt(top_k=top_k)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        for iteration in range(max_iterations):
            is_last = iteration == max_iterations - 1
            _log.info("Agent iteration %d/%d", iteration + 1, max_iterations)

            assistant_msg = await _call_openai_async(
                client, cfg.agent_llm_model, messages, self._tool_schemas,
            )
            if assistant_msg is None:
                _log.warning("LLM call failed at iteration %d, falling back", iteration + 1)
                return self._search.search_files(query, top_k=top_k)

            messages.append(assistant_msg)

            tool_calls = assistant_msg.get("tool_calls") or []

            # --- Natural termination: no tool calls = agent is done ---
            if not tool_calls:
                content = assistant_msg.get("content", "")
                if content:
                    _log.info("Agent stopped naturally (no tool calls): %s", content[:200])
                return self._build_results_from_history(query, messages, top_k=top_k)

            # Process tool calls (parallel-safe tools run concurrently)
            tool_results, extracted_symbols = self._dispatch_tool_calls(
                tool_calls, default_top_k=top_k,
            )
            messages.extend(tool_results)

            # Inject extracted symbols as a hint for the LLM
            if extracted_symbols:
                unique = list(dict.fromkeys(extracted_symbols))[:15]
                hint = (
                    f"[Hint] Symbols found in the code you just read: {unique}. "
                    "Consider searching for these exact names with search_code, "
                    "and use list_related_files on the files you read."
                )
                messages.append({"role": "user", "content": hint})
                _log.info("Injected symbol hints: %s", unique)

            # --- Last iteration: force extract from history ---
            if is_last:
                _log.info("Agent reached max iterations, extracting results from history")
                return self._build_results_from_history(query, messages, top_k=top_k)

        # Should not reach here, but safety fallback
        return self._search.search_files(query, top_k=top_k)

    def run_sync(self, query: str, max_iterations: int = 5, top_k: int = 10) -> list[FileSearchResult]:
        """Synchronous wrapper around ``run()`` for CLI callers.

        Uses ``asyncio.run()`` to execute the async agent loop.
        """
        return asyncio.run(self.run(query, max_iterations=max_iterations, top_k=top_k))

    def _build_system_prompt(self, *, top_k: int) -> str:
        return (
            "You are a code localization agent. Your goal is to find ALL source files "
            "that need to be modified to fix a bug or implement a feature.\n\n"
            "You MUST follow this workflow strictly:\n\n"
            "**Step 1 - Initial Search**: Use `search_code` with keywords from the "
            "user's request (error messages, function names, concepts).\n\n"
            "**Step 2 - Read Top Files**: Use `read_files_batch` to read the top 3-5 "
            "files from search results. Do NOT assume the first result is the only answer.\n\n"
            "**Step 3 - Extract Symbols & Deepen Search**: From the code you read, "
            "identify specific function/class names. Use these EXACT symbol names in a "
            "NEW `search_code` call. For example, if you see `result = segment2box(seg)`, "
            "search for `segment2box`. Do NOT just search with natural language.\n\n"
            "**Step 4 - Explore Dependencies**: For each highly relevant file, ALWAYS "
            "call `list_related_files` to discover its imports, callers, and callees. "
            "Bug fixes often require changes in MULTIPLE related files.\n\n"
            "**Step 5 - Terminate**: When confident, stop calling tools and respond with "
            "text listing the file paths ranked by relevance.\n\n"
            "CRITICAL RULES:\n"
            "- If file_A imports from file_B, BOTH may need changes. Use `list_related_files`.\n"
            "- Do NOT spend multiple iterations reading the same file. Move on to explore.\n"
            "- After reading code, search for the specific symbols you found, not paraphrases.\n"
            "- You can call multiple tools in a single response. When you need to "
            "read multiple files, use separate read_files_batch or get_entity_content "
            "calls — they execute in parallel.\n\n"
            f"Return at most {top_k} files, ranked by relevance.\n"
        )

    # -----------------------------------------------------------------------
    # Graph-enhanced mode
    # -----------------------------------------------------------------------

    def _build_analysis_system_prompt(self) -> str:
        return (
            "You are a code relationship analysis agent. Your goal is NOT to list files -- "
            "a separate search pipeline handles retrieval and ranking.\n\n"
            "Your job is to DISCOVER HIDDEN RELATIONSHIPS between code entities that "
            "static analysis might miss. Focus on:\n\n"
            "**Step 1 - Initial Search**: Use `search_code` to find entry points related to the query.\n\n"
            "**Step 2 - Read & Analyze**: Use `read_files_batch` to understand the code deeply.\n\n"
            "**Step 3 - Report Relationships**: For EVERY relationship you discover, call "
            "`report_relationship`. Types:\n"
            "  - `import`: A imports/uses B\n"
            "  - `call`: A calls functions in B\n"
            "  - `inherit`: A extends/implements B\n"
            "  - `co_change`: A and B must be modified together (e.g., interface + implementation)\n"
            "  - `semantic`: semantically related (e.g., both handle auth)\n\n"
            "**Step 4 - Follow Chains**: Use `list_related_files` to explore dependency chains. "
            "Report each discovered link with `report_relationship`.\n\n"
            "CRITICAL RULES:\n"
            "- Call `report_relationship` for EVERY pair of related files you find\n"
            "- Higher confidence (0.9-1.0) for direct imports/calls you verified in code\n"
            "- Lower confidence (0.5-0.7) for semantic/co_change relationships\n"
            "- Aim for 5-15 relationships per query\n"
            "- When done analyzing, stop calling tools\n"
        )

    def _inject_edges_to_graph(self, edges: list[dict[str, Any]]) -> int:
        """Inject agent-discovered edges into the entity graph. Returns count of edges added."""
        if not self._entity_graph or not edges:
            return 0

        from codexlens_search.core.entity_graph import _entity_for_file

        added = 0
        for edge in edges:
            from_file = edge.get("from_file", "")
            to_file = edge.get("to_file", "")
            kind = edge.get("kind", "semantic")
            confidence = float(edge.get("confidence", 0.8))
            if not from_file or not to_file:
                continue
            from_ent = _entity_for_file(from_file)
            to_ent = _entity_for_file(to_file)
            self._entity_graph.add_edge(from_ent, to_ent, kind, weight=confidence)
            added += 1

        _log.info("Injected %d agent-discovered edges into entity graph", added)
        return added

    def _extract_edges_from_messages(
        self, messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Fallback: auto-extract file relationships from search results and code reads.

        When the LLM never calls report_relationship, mine the conversation
        history for file co-occurrence and import relationships.
        """
        # Collect all file paths seen in search results
        search_files: list[str] = _extract_paths_from_search_results(messages)

        # Collect all files the agent chose to read (strong relevance signal)
        read_files: list[str] = []
        for msg in messages:
            for tc in msg.get("tool_calls", []):
                fn = tc.get("function", {})
                name = fn.get("name", "")
                if name in ("read_files_batch", "get_entity_content"):
                    raw = fn.get("arguments", "{}")
                    try:
                        args = json.loads(raw) if isinstance(raw, str) else raw
                    except Exception:
                        continue
                    if name == "read_files_batch":
                        for fp in (args.get("file_paths") or []):
                            if fp and fp not in read_files:
                                read_files.append(str(fp))
                    elif name == "get_entity_content":
                        fp = args.get("file_path", "")
                        if fp and fp not in read_files:
                            read_files.append(str(fp))

        # Extract import relationships from code that was read
        edges: list[dict[str, Any]] = []
        seen_pairs: set[tuple[str, str]] = set()
        cwd = Path(os.getcwd())

        for msg in messages:
            if msg.get("role") != "tool":
                continue
            content = msg.get("content", "") or ""
            if not content or len(content) < 20:
                continue

            # Find which file this tool result corresponds to
            source_file = ""
            for fp in read_files:
                # Normalize for comparison
                fp_norm = fp.replace("\\", "/")
                if fp_norm in content or fp.replace("/", "\\") in content:
                    source_file = fp_norm
                    break

            if not source_file:
                continue

            # Extract imports from the code
            for mod in _extract_imports_from_text(content):
                target = _resolve_module_to_path(mod, cwd)
                if not target:
                    continue
                pair = (source_file, target)
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    edges.append({
                        "from_file": source_file,
                        "to_file": target,
                        "kind": "import",
                        "confidence": 0.9,
                    })

        # Co-occurrence: files that appear together in search results → semantic relationship
        if len(search_files) >= 2:
            top_search = search_files[:8]
            for i, f1 in enumerate(top_search):
                for f2 in top_search[i + 1:]:
                    f1n = f1.replace("\\", "/")
                    f2n = f2.replace("\\", "/")
                    pair = (f1n, f2n)
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        edges.append({
                            "from_file": f1n,
                            "to_file": f2n,
                            "kind": "co_change",
                            "confidence": 0.6,
                        })

        _log.info("Fallback edge extraction: %d edges from %d search files + %d read files",
                  len(edges), len(search_files), len(read_files))
        return edges

    async def _run_analysis_loop(
        self,
        query: str,
        max_iterations: int,
        client: Any,
    ) -> list[dict[str, Any]]:
        """Run agent analysis loop focused on discovering relationships.

        Returns list of discovered edge dicts.
        """
        self._discovered_edges = []  # Reset accumulator

        tool_schemas = get_graph_enhanced_tool_schemas()
        system_prompt = self._build_analysis_system_prompt()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        cfg = self._config
        for iteration in range(max_iterations):
            is_last = iteration == max_iterations - 1
            _log.info("Analysis agent iteration %d/%d", iteration + 1, max_iterations)

            assistant_msg = await _call_openai_async(
                client, cfg.agent_llm_model, messages, tool_schemas,
            )
            if assistant_msg is None:
                _log.warning("Analysis LLM call failed at iteration %d", iteration + 1)
                break

            messages.append(assistant_msg)
            tool_calls = assistant_msg.get("tool_calls") or []

            if not tool_calls:
                _log.info("Analysis agent stopped naturally, discovered %d edges", len(self._discovered_edges))
                break

            # Process tool calls
            tool_results, _ = self._dispatch_tool_calls(
                tool_calls, default_top_k=10,
            )
            messages.extend(tool_results)

            # Inject hint to nudge LLM toward report_relationship
            has_report = any(
                (tc.get("function") or {}).get("name") == "report_relationship"
                for tc in tool_calls
            )
            if not has_report and not is_last:
                hint = (
                    "[IMPORTANT] You have read code but not yet called `report_relationship`. "
                    "For every file pair you discovered that are related (imports, calls, "
                    "co-change), call `report_relationship` NOW. "
                    "Do NOT just search — your primary job is to REPORT RELATIONSHIPS."
                )
                messages.append({"role": "user", "content": hint})

        edges = list(self._discovered_edges)
        self._discovered_edges = []  # Clean up

        # Fallback: if LLM never called report_relationship, extract from history
        if not edges:
            _log.info("LLM reported 0 edges, running fallback extraction from conversation")
            edges = self._extract_edges_from_messages(messages)

        return edges

    async def _run_graph_enhanced(
        self, query: str, max_iterations: int, top_k: int,
    ) -> list[FileSearchResult]:
        """Graph-enhanced mode: Pipeline + Agent parallel, Agent enriches graph."""
        cfg = self._config

        client = self._get_async_client()
        if client is None:
            _log.warning("No async LLM client, falling back to pipeline search")
            return self._search.search_files(query, top_k=top_k)

        # Phase 1: Pipeline search + Agent analysis in parallel
        pipeline_task = asyncio.to_thread(
            self._search.search_files, query, top_k=top_k * 3,
        )
        agent_task = self._run_analysis_loop(query, max_iterations, client)

        pipeline_results, agent_edges = await asyncio.gather(
            pipeline_task, agent_task, return_exceptions=True,
        )

        # Handle errors
        if isinstance(pipeline_results, Exception):
            _log.warning("Pipeline search failed: %s", pipeline_results)
            pipeline_results = []
        if isinstance(agent_edges, Exception):
            _log.warning("Agent analysis failed: %s", agent_edges)
            agent_edges = []

        if not agent_edges:
            _log.info("Agent found no new edges, returning pipeline results directly")
            return pipeline_results[:top_k]

        # Phase 2: Inject agent-discovered edges into graph
        injected = self._inject_edges_to_graph(agent_edges)
        _log.info("Graph enhanced with %d new edges from agent", injected)

        if injected == 0:
            return pipeline_results[:top_k]

        # Phase 3: Re-run pipeline search with enriched graph
        enhanced_results = self._search.search_files(query, top_k=top_k)

        _log.info(
            "Graph-enhanced search: pipeline=%d initial, agent=%d edges, enhanced=%d final",
            len(pipeline_results), injected, len(enhanced_results),
        )

        return enhanced_results

    async def _run_hybrid(
        self, query: str, max_iterations: int, top_k: int,
    ) -> list[FileSearchResult]:
        """Hybrid mode: full agent loop (iterative discovery) + graph enhancement + pipeline reranking.

        Combines the best of agent mode (multi-round iterative file discovery)
        with graph enhancement (agent reports relationships) and pipeline
        reranking (proper scoring via reranker/FTS/fusion).

        Flow:
        1. Run full agent loop with graph-enhanced tools (includes report_relationship)
        2. Collect agent-discovered files AND edges from conversation
        3. Inject edges into entity graph
        4. Merge agent-discovered files with pipeline search results
        5. Use pipeline reranker for final scoring
        """
        cfg = self._config

        client = self._get_async_client()
        if client is None:
            _log.warning("No async LLM client, falling back to pipeline search")
            return self._search.search_files(query, top_k=top_k)

        # Phase 1: Run the full agent loop with graph-enhanced tool schemas
        # (same as _run_single but with report_relationship tool available)
        self._discovered_edges = []
        tool_schemas = get_graph_enhanced_tool_schemas()
        system_prompt = self._build_system_prompt(top_k=top_k)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        for iteration in range(max_iterations):
            is_last = iteration == max_iterations - 1
            _log.info("Hybrid agent iteration %d/%d", iteration + 1, max_iterations)

            assistant_msg = await _call_openai_async(
                client, cfg.agent_llm_model, messages, tool_schemas,
            )
            if assistant_msg is None:
                _log.warning("LLM call failed at iteration %d, falling back", iteration + 1)
                break

            messages.append(assistant_msg)
            tool_calls = assistant_msg.get("tool_calls") or []

            if not tool_calls:
                _log.info("Hybrid agent stopped naturally: %s",
                          (assistant_msg.get("content") or "")[:200])
                break

            tool_results, extracted_symbols = self._dispatch_tool_calls(
                tool_calls, default_top_k=top_k,
            )
            messages.extend(tool_results)

            if extracted_symbols:
                unique = list(dict.fromkeys(extracted_symbols))[:15]
                hint = (
                    f"[Hint] Symbols found in the code you just read: {unique}. "
                    "Consider searching for these exact names with search_code, "
                    "and use list_related_files on the files you read."
                )
                messages.append({"role": "user", "content": hint})

            if is_last:
                _log.info("Hybrid agent reached max iterations")

        # Phase 2: Collect agent-discovered files from conversation history
        agent_files = _extract_paths_from_search_results(messages)
        text_files = _extract_file_paths_from_messages(messages)
        seen: set[str] = set()
        agent_discovered: list[str] = []
        for p in agent_files + text_files:
            if p not in seen:
                seen.add(p)
                agent_discovered.append(p)

        # Phase 3: Collect edges (explicit report_relationship + fallback extraction)
        edges = list(self._discovered_edges)
        self._discovered_edges = []
        if not edges:
            edges = self._extract_edges_from_messages(messages)

        # Phase 4: Inject edges into entity graph
        if edges:
            injected = self._inject_edges_to_graph(edges)
            _log.info("Hybrid: injected %d edges into graph", injected)

        # Phase 5: Pipeline search (with enriched graph)
        pipeline_results = self._search.search_files(query, top_k=max(top_k * 3, 30))
        pipeline_by_path = {r.path: r for r in pipeline_results}

        # Phase 6: Merge agent-discovered files with pipeline results
        # Agent files preserve their discovery order (reflecting iterative refinement).
        # Pipeline provides proper scores via reranker.
        out: list[FileSearchResult] = []
        used: set[str] = set()

        for p in agent_discovered:
            if len(out) >= top_k:
                break
            if p in used:
                continue
            used.add(p)
            if p in pipeline_by_path:
                out.append(pipeline_by_path[p])
            else:
                # Agent found a file pipeline missed — include with score 0
                out.append(FileSearchResult(
                    path=str(p), score=0.0, best_chunk_id=0,
                    snippet="", line=0, end_line=0, content="", language="",
                    chunk_ids=(),
                ))

        # Fill remaining slots from pipeline (sorted by score)
        for r in pipeline_results:
            if len(out) >= top_k:
                break
            if r.path not in used:
                used.add(r.path)
                out.append(r)

        _log.info(
            "Hybrid result: agent_discovered=%d, edges=%d, pipeline=%d, final=%d",
            len(agent_discovered), len(edges), len(pipeline_results), len(out),
        )
        return out

    def _execute_tool_call(
        self, tc: dict[str, Any], default_top_k: int,
    ) -> tuple[dict[str, Any], list[str]]:
        """Execute a single tool call and return (tool_result_message, extracted_symbols)."""
        tool_call_id = str(tc.get("id") or "")
        fn = tc.get("function") or {}
        name = str(fn.get("name") or "")
        raw_args = fn.get("arguments") or "{}"
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
        except Exception:
            args = {}

        _log.info("Agent tool call: %s(%s)", name, json.dumps(args, ensure_ascii=False)[:200])

        result = self._execute_tool(name, args, default_top_k=default_top_k)
        msg: dict[str, Any] = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": result,
        }

        symbols: list[str] = []
        if name in ("read_files_batch", "get_entity_content"):
            symbols = _extract_symbols_from_text(result)

        return msg, symbols

    def _dispatch_tool_calls(
        self,
        tool_calls: list[dict[str, Any]],
        *,
        default_top_k: int,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Dispatch tool calls, running allowlisted ones in parallel.

        Returns (ordered_tool_result_messages, all_extracted_symbols).
        """
        cfg = self._config
        concurrency = cfg.agent_tool_concurrency
        allowlist = set(cfg.agent_parallel_tools_allowlist)

        # Partition into parallel-safe and serial groups, preserving original index
        parallel_indices: list[int] = []
        serial_indices: list[int] = []
        for i, tc in enumerate(tool_calls):
            fn = tc.get("function") or {}
            name = str(fn.get("name") or "")
            if concurrency > 1 and name in allowlist:
                parallel_indices.append(i)
            else:
                serial_indices.append(i)

        # Pre-allocate result slots
        results: list[tuple[dict[str, Any], list[str]] | None] = [None] * len(tool_calls)

        # Execute parallel group
        if parallel_indices and concurrency > 1:
            workers = min(concurrency, len(parallel_indices))
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        self._execute_tool_call, tool_calls[i], default_top_k,
                    ): i
                    for i in parallel_indices
                }
                for future in futures:
                    idx = futures[future]
                    results[idx] = future.result()
        else:
            # Fallback: run "parallel" group serially when concurrency=1
            for i in parallel_indices:
                results[i] = self._execute_tool_call(tool_calls[i], default_top_k)

        # Execute serial group sequentially
        for i in serial_indices:
            results[i] = self._execute_tool_call(tool_calls[i], default_top_k)

        # Merge in original order
        ordered_msgs: list[dict[str, Any]] = []
        all_symbols: list[str] = []
        for r in results:
            assert r is not None
            msg, symbols = r
            ordered_msgs.append(msg)
            all_symbols.extend(symbols)

        return ordered_msgs, all_symbols

    def _execute_tool(self, name: str, args: dict[str, Any], *, default_top_k: int) -> str:
        if name == "search_code":
            return self._tool_search_code(args, default_top_k=default_top_k)

        if name == "traverse_graph":
            seed = str(args.get("entity_name") or "")
            depth = max(0, _safe_int(args.get("depth"), self._config.entity_graph_depth))
            return json.dumps(self._traverse_entities(seed, depth=depth), ensure_ascii=False)

        if name == "get_entity_content":
            file_path = str(args.get("file_path") or "")
            start_line = max(1, _safe_int(args.get("start_line"), 1))
            end_line = max(start_line, _safe_int(args.get("end_line"), start_line + 200))
            return self._read_file_range(file_path, start_line=start_line, end_line=end_line)

        if name == "read_files_batch":
            return self._tool_read_files_batch(args)

        if name == "list_related_files":
            return self._tool_list_related_files(args)

        if name == "report_relationship":
            from_file = str(args.get("from_file") or "")
            to_file = str(args.get("to_file") or "")
            kind = str(args.get("kind") or "semantic")
            confidence = float(args.get("confidence", 0.8))
            if from_file and to_file:
                edge = {"from_file": from_file, "to_file": to_file, "kind": kind, "confidence": confidence}
                self._discovered_edges.append(edge)
                _log.info("Agent reported relationship: %s -[%s]-> %s (%.2f)", from_file, kind, to_file, confidence)
                return json.dumps({"status": "recorded", "edge": edge})
            return json.dumps({"error": "from_file and to_file are required"})

        return json.dumps({"error": f"unknown tool: {name}"})

    # -----------------------------------------------------------------------
    # Tool implementations
    # -----------------------------------------------------------------------

    def _tool_search_code(self, args: dict[str, Any], *, default_top_k: int) -> str:
        query = str(args.get("query") or "")
        mode = str(args.get("mode") or "thorough")
        top_k = max(1, _safe_int(args.get("top_k"), default_top_k))
        results = self._search.search(query, top_k=top_k, quality=mode)
        payload = [
            {
                "id": r.id,
                "path": r.path,
                "score": float(r.score),
                "line": int(r.line),
                "end_line": int(r.end_line),
                "snippet": r.snippet,
            }
            for r in results
        ]
        return json.dumps(payload, ensure_ascii=False)

    def _tool_read_files_batch(self, args: dict[str, Any]) -> str:
        file_paths = list(args.get("file_paths") or [])
        max_lines = max(10, min(200, _safe_int(args.get("max_lines_per_file"), 50)))
        results: list[dict[str, Any]] = []

        for fp in file_paths[:20]:
            fp = str(fp)
            content = self._read_file_range(fp, start_line=1, end_line=max_lines)
            if content.startswith("error:"):
                results.append({"path": fp, "error": content})
            else:
                results.append({"path": fp, "content": content})

        return json.dumps(results, ensure_ascii=False)

    def _tool_list_related_files(self, args: dict[str, Any]) -> str:
        seed = str(args.get("entity_name") or "")
        depth = max(1, min(3, _safe_int(args.get("depth"), 2)))

        entities = self._traverse_entities(seed, depth=depth)
        if not entities:
            return json.dumps({"files": [], "message": "No related entities found"})

        # Deduplicate by file_path, keep relationship info
        seen: set[str] = set()
        files: list[dict[str, str]] = []
        for ent in entities:
            fp = ent.get("file_path", "")
            if fp and fp not in seen:
                seen.add(fp)
                files.append({
                    "path": fp,
                    "symbol": ent.get("symbol_name", ""),
                    "kind": ent.get("kind", ""),
                })

        return json.dumps({"files": files, "count": len(files)}, ensure_ascii=False)

    # -----------------------------------------------------------------------
    # Entity graph traversal
    # -----------------------------------------------------------------------

    def _traverse_entities(self, seed: str, *, depth: int) -> list[dict[str, Any]]:
        if not seed or self._entity_graph is None or not self._config.entity_graph_enabled:
            return []

        seeds: list[EntityId] = []
        if _is_path_like(seed):
            seeds.append(
                EntityId(
                    file_path=seed,
                    symbol_name="",
                    kind=EntityKind.FILE,
                    start_line=0,
                    end_line=0,
                )
            )
        else:
            syms = self._search._fts.get_symbols_by_name(seed)
            for sym in syms[:5]:
                cid = sym.get("chunk_id")
                if cid is None:
                    continue
                path = self._search._fts.get_doc_meta(int(cid))[0]
                try:
                    kind = EntityKind(str(sym.get("kind", "")).lower())
                except Exception:
                    continue
                seeds.append(
                    EntityId(
                        file_path=str(path),
                        symbol_name=str(sym.get("name", "")),
                        kind=kind,
                        start_line=int(sym.get("start_line", 0) or 0),
                        end_line=int(sym.get("end_line", 0) or 0),
                    )
                )

        if not seeds:
            return []

        seen: set[str] = set()
        out: list[dict[str, Any]] = []
        for s in seeds:
            for ent in self._entity_graph.traverse(s, depth=depth):
                key = ent.to_key()
                if key in seen:
                    continue
                seen.add(key)
                out.append(
                    {
                        "file_path": ent.file_path,
                        "symbol_name": ent.symbol_name,
                        "kind": ent.kind.value,
                        "start_line": ent.start_line,
                        "end_line": ent.end_line,
                    }
                )
        return out

    # -----------------------------------------------------------------------
    # File reading
    # -----------------------------------------------------------------------

    def _read_file_range(self, file_path: str, *, start_line: int, end_line: int) -> str:
        if not file_path:
            return "error: empty file_path"

        candidates: list[Path] = []
        p = Path(file_path)
        candidates.append(p)
        if not p.is_absolute():
            candidates.append(Path(os.getcwd()) / p)

        resolved: Path | None = None
        for c in candidates:
            if c.exists() and c.is_file():
                resolved = c
                break

        if resolved is None:
            return f"error: file not found: {file_path}"

        try:
            lines = resolved.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception as e:
            return f"error: failed to read file: {file_path}: {e}"

        start_idx = max(0, min(len(lines), start_line - 1))
        end_idx = max(start_idx, min(len(lines), end_line))
        snippet = lines[start_idx:end_idx]

        out_lines = []
        for i, text in enumerate(snippet, start=start_idx + 1):
            out_lines.append(f"{i}: {text}")
        return "\n".join(out_lines)

    # -----------------------------------------------------------------------
    # Result building from conversation history
    # -----------------------------------------------------------------------

    def _build_results_from_history(
        self,
        query: str,
        messages: list[dict[str, Any]],
        *,
        top_k: int,
    ) -> list[FileSearchResult]:
        """Extract file paths from conversation history and build results.

        Priority: paths from search results > paths from text mentions.
        Falls back to plain search_files if no paths found.
        """
        # Collect paths from search_code tool results (highest confidence)
        search_paths = _extract_paths_from_search_results(messages)
        # Collect paths from all text mentions
        text_paths = _extract_file_paths_from_messages(messages)

        # Merge: search paths first (ordered by search score), then text paths
        seen: set[str] = set()
        merged: list[str] = []
        for p in search_paths + text_paths:
            if p not in seen:
                seen.add(p)
                merged.append(p)

        if not merged:
            _log.info("No file paths extracted from history, falling back to search_files")
            return self._search.search_files(query, top_k=top_k)

        _log.info("Extracted %d file paths from agent history: %s", len(merged), merged[:10])

        # --- Import-aware auto-expansion ---
        # For top-ranked files, parse their imports and interleave project-local
        # dependencies right after the parent file (not appended at end).
        cwd = Path(os.getcwd())
        expanded: list[str] = []
        expand_count = 0
        for p in merged:
            expanded.append(p)
            if expand_count >= 5:
                continue
            expand_count += 1
            content = self._read_file_range(p, start_line=1, end_line=500)
            if content.startswith("error:"):
                continue
            for mod in _extract_imports_from_text(content):
                rp = _resolve_module_to_path(mod, cwd)
                if rp and rp not in seen:
                    seen.add(rp)
                    expanded.append(rp)
            if len(expanded) > len(merged):
                _log.info("Import expansion for %s added files, total now %d", p, len(expanded))
        merged = expanded

        # Get baseline scored results
        baseline = self._search.search_files(query, top_k=max(top_k * 5, top_k))
        by_path = {r.path: r for r in baseline}

        out: list[FileSearchResult] = []
        for p in merged:
            if len(out) >= top_k:
                break
            if p in by_path:
                out.append(by_path[p])
            else:
                out.append(
                    FileSearchResult(
                        path=str(p),
                        score=0.0,
                        best_chunk_id=0,
                        snippet="",
                        line=0,
                        end_line=0,
                        content="",
                        language="",
                        chunk_ids=(),
                    )
                )
        return out
