from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from codexlens_search.agent.tools import get_tool_schemas
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
        self._client: Any = None  # lazy init

    def _get_client(self) -> Any | None:
        if self._client is None:
            cfg = self._config
            self._client = _create_openai_client(cfg.agent_llm_api_key, cfg.agent_llm_api_base)
        return self._client

    def run(self, query: str, max_iterations: int = 5, top_k: int = 10) -> list[FileSearchResult]:
        cfg = self._config
        max_iterations = max(1, _safe_int(max_iterations, 5))
        top_k = max(1, _safe_int(top_k, 10))

        if not cfg.agent_enabled:
            _log.debug("Agent disabled, falling back to search_files")
            return self._search.search_files(query, top_k=top_k)

        client = self._get_client()
        if client is None:
            _log.warning("No LLM client available, falling back to search_files")
            return self._search.search_files(query, top_k=top_k)

        system_prompt = self._build_system_prompt(top_k=top_k)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        for iteration in range(max_iterations):
            is_last = iteration == max_iterations - 1
            _log.info("Agent iteration %d/%d", iteration + 1, max_iterations)

            assistant_msg = _call_openai(
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

            # Process tool calls
            extracted_symbols: list[str] = []
            for tc in tool_calls:
                tool_call_id = str(tc.get("id") or "")
                fn = tc.get("function") or {}
                name = str(fn.get("name") or "")
                raw_args = fn.get("arguments") or "{}"
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
                except Exception:
                    args = {}

                _log.info("Agent tool call: %s(%s)", name, json.dumps(args, ensure_ascii=False)[:200])

                result = self._execute_tool(name, args, default_top_k=top_k)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": result,
                })

                # --- Symbol extraction after file reading ---
                if name in ("read_files_batch", "get_entity_content"):
                    extracted_symbols.extend(_extract_symbols_from_text(result))

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
            "- After reading code, search for the specific symbols you found, not paraphrases.\n\n"
            f"Return at most {top_k} files, ranked by relevance.\n"
        )

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
