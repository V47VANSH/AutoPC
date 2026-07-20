"""
One-shot model calls for the agent's off-hot-path thinking.

The driver runs in a long-lived SDK session with desktop tools. Everything else
the agent "thinks" about -- turning a goal into a definition of done, checking a
completion claim against the screen, reflecting on a failed attempt, distilling
a skill -- is a single, self-contained question. Those go through here:

    generate_text(prompt, system, images=...)          -> str
    generate_json(prompt, schema, system, images=...)  -> dict

Each call is a fresh `query()` with no tools and its own system prompt, so it
never touches (or is biased by) the driver's conversation. That isolation is
the point for verification: a judge that shares the actor's context inherits
its blind spots.

`generate_json` uses the SDK's structured output (`output_format` json_schema),
which validates against the schema; a lenient text parse is the fallback.
Images are sent as Messages-API image blocks in a stream-json user message,
verified to reach the model (a probe image's content came back verbatim).
"""
from __future__ import annotations

import base64
import io
import json
import re
from typing import Any, Dict, List, Optional

from claude_agent_sdk import (
    AssistantMessage, ClaudeAgentOptions, ResultMessage, TextBlock, query,
)

from autopc import config


class LLMError(RuntimeError):
    """The model could not produce a usable answer."""


def default_model() -> str:
    """The reasoning model for off-hot-path work (contract, verify, reflect)."""
    return config.solver_model()


#: The API downsamples images whose long edge exceeds ~1568px, so sending more
#: is wasted upload. Text on a 1920px screen stays legible at this size.
MAX_IMAGE_EDGE = 1568


def image_block(image, max_edge: int = MAX_IMAGE_EDGE) -> Dict[str, Any]:
    """PIL image -> a Messages-API image block. Lossless WebP when available
    (smallest exact encoding for UI screenshots), PNG otherwise."""
    img = image.convert("RGB")
    w, h = img.size
    scale = min(1.0, max_edge / float(max(w, h)))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)))
    buf = io.BytesIO()
    try:
        img.save(buf, format="WEBP", lossless=True, method=4)
        media = "image/webp"
    except Exception:
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        media = "image/png"
    return {"type": "image", "source": {"type": "base64", "media_type": media,
                                        "data": base64.b64encode(buf.getvalue()).decode("ascii")}}


def _options(model: Optional[str], system: str, max_turns: int,
             output_format: Optional[dict] = None) -> ClaudeAgentOptions:
    return ClaudeAgentOptions(
        model=model or default_model(),
        system_prompt=system or "You are a careful, precise assistant.",
        mcp_servers={}, tools=[], allowed_tools=[],
        permission_mode="bypassPermissions", setting_sources=[],
        max_turns=max_turns, max_buffer_size=16 * 1024 * 1024,
        output_format=output_format,
    )


async def _user_message(text: str, images: Optional[List[Any]]):
    content: List[Dict[str, Any]] = [image_block(im) for im in (images or [])]
    content.append({"type": "text", "text": text})
    yield {"type": "user", "message": {"role": "user", "content": content},
           "parent_tool_use_id": None}


async def _run(prompt: str, system: str, model: Optional[str], images, max_turns: int,
               output_format: Optional[dict]):
    text_chunks: List[str] = []
    result: Optional[ResultMessage] = None
    stream_prompt: Any = _user_message(prompt, images) if images else prompt
    async for message in query(prompt=stream_prompt,
                               options=_options(model, system, max_turns, output_format)):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    text_chunks.append(block.text)
        elif isinstance(message, ResultMessage):
            result = message
    return "".join(text_chunks).strip(), result


async def generate_text(prompt: str, system: str = "", model: Optional[str] = None,
                        images: Optional[List[Any]] = None) -> str:
    text, result = await _run(prompt, system, model, images, max_turns=1, output_format=None)
    if not text and result is not None and result.result:
        text = result.result.strip()
    if not text:
        raise LLMError("empty response")
    return text


_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Lenient JSON extraction: a fenced block, or the outermost {...}."""
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidates = [fenced.group(1)] if fenced else []
    m = _JSON_OBJECT.search(text)
    if m:
        candidates.append(m.group(0))
    for c in candidates:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj
        except ValueError:
            continue
    return None


async def generate_json(prompt: str, schema: Dict[str, Any], system: str = "",
                        model: Optional[str] = None,
                        images: Optional[List[Any]] = None) -> Dict[str, Any]:
    """Ask for a JSON object matching `schema`. Structured output needs a second
    internal turn to emit the result, hence max_turns=3."""
    text, result = await _run(prompt, system, model, images, max_turns=3,
                              output_format={"type": "json_schema", "schema": schema})
    if result is not None and isinstance(result.structured_output, dict):
        return result.structured_output
    for candidate in (text, getattr(result, "result", None) or ""):
        obj = extract_json(candidate)
        if obj is not None:
            return obj
    raise LLMError("no JSON object in response")
