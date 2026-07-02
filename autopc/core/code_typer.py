"""
Typing generated code into a real editor without the editor corrupting it.

## The problem

`hands.type_text` sends every character of the model's code through the
Interception keyboard, including real Enter and Tab. A plain text box takes
that verbatim. A **code editor does not** -- it "helps", and every helpful
feature mangles code that already contains its own indentation and brackets:

  1. **Auto-closing pairs.** Typing `(` `[` `{` `"` `'` inserts the closer too.
     When the model then types its own closer, you get `print())`, `""`, etc.
  2. **Auto-indent on Enter.** After a newline the editor injects leading
     whitespace; the model's line carries its own indent as well, so the two
     stack into doubled, cascading indentation that worsens with depth.
  3. **Bracket-split on Enter.** Pressing Enter inside an auto-closed `{}` pushes
     the `}` onto its own indented line -- so the model's own `}` later becomes a
     phantom extra line.

Pasting (clipboard + Ctrl+V) sidesteps all of this, but it is not always
available or reliable -- some editors intercept or reformat a paste, and some
environments disable it. So we must keep *typing* -- character by character --
and make the typing survive the editor.

## The invariant that makes it deterministic

Type at the **end of the document** (an empty or just-cleared editor -- the
"replace the boilerplate" flow). Then the cursor is always at end-of-content
with nothing to its right and no line below it. Under that invariant a forward
`Delete` can only ever do one of two things: remove an auto-inserted closer
sitting immediately to the right, or nothing at all. It can never eat a real
character or merge a line. That is what lets us neutralize the quirks with a
few extra keystrokes and no fragile per-editor detection:

  * **Auto-close:** after typing each opener, tap `Delete`. If the editor
    auto-closed, the closer is removed; if it did not, the Delete is a no-op.
    Either way we then type the model's own closer literally and it lands right.
  * **Auto-indent:** handled by one of two strategies -- see below.

Nothing else is sent between lines: a newline is a bare `Enter`, exactly as a
person types it. An earlier version tapped `Escape` before every `Enter` to close
any open IntelliSense popup (which would otherwise swallow the Enter as "accept
suggestion"). That cost one Escape *per line* -- dozens per solution -- and
Escape is not a neutral key: it exits browser/app full-screen, cancels an
in-flight page load and closes modals, so a long typing run could knock a
full-screened app out of full screen. Losing the guard risks only the
suggestion case, and only in editors that both auto-complete aggressively and
bind Enter to accept; that is the cheaper failure.

## Two indent strategies: fight the editor, or ride it

Neutralizing auto-indent by brute force works, but it is the slow, obviously
robotic part of the job. For every single line it costs `Shift+Home`, `Delete`,
and then re-typing the whole indent one space at a time -- on a 45-line Java
solution that is ~180 wasted keystrokes and the bulk of the wall-clock time. It
is also nothing like how a person writes code: nobody clears the editor's indent
and re-types eight spaces. They press Enter and start typing the statement,
because the editor already put them at the right column.

So there are two modes, picked automatically per snippet:

**`editor` -- ride the auto-indent (brace-delimited languages).** Editors follow
one well-known rule: after Enter, indent the new line like the previous line,
plus one unit if the previous line ended with an opener (`{`, `(`, `[`, `:`),
and re-indent the line when you type a closer (`}`, `)`, `]`) as its first
character. That is Monaco's / VS Code's `increaseIndentPattern` /
`decreaseIndentPattern`, and CodeMirror behaves the same -- between them that is
essentially every browser-based code editor. So we *predict* what the editor will
do and type only the difference: usually nothing at all, occasionally a few
spaces. No Shift+Home, no Delete, no re-typed indentation.

Predictions can be wrong -- a plain `<textarea>` does not auto-indent at all, and
an editor may indent continuation lines its own way. In a brace-delimited
language that costs *only cosmetics*: the code compiles and runs identically
whatever column it sits in. That is exactly why this mode is restricted to those
languages, and why the second mode exists.

**`exact` -- clear and re-type (whitespace-significant code).** In Python, YAML,
a Java text block or a JS template literal, leading whitespace *is* meaning, so a
mispredicted column is a wrong answer. There we keep the deterministic path:
after each `Enter`, `Shift+Home` selects whatever was auto-indented and `Delete`
removes it, leaving a clean line onto which we type the exact indentation.
(Deleting the selection *before* typing also avoids the editor's "surround
selection with brackets" behaviour for a line that begins with `(` or a quote.)
The indent itself is typed as a fast burst rather than at prose cadence -- a
person hits Tab or holds space; they do not type eight spaces individually.

`indent_is_significant()` makes the choice, from an explicit language hint when
the caller has one and otherwise by sniffing the source. It errs towards `exact`,
which is never wrong -- only slower.

Everything is emitted through the normal human-timed `hands` path -- ordinary
keystrokes that simply never fight the editor.

## Scope

v1 targets typing a whole code block into an empty/cleared editor (the common
case). Inserting into the *middle* of existing code breaks the
end-of-document invariant (there a forward Delete could merge lines), so that is
intentionally out of scope here.
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

#: Opening brackets whose auto-inserted partner we delete after typing them.
OPENERS = "([{"
#: Quote characters. They open *and* close, but at end-of-content the delete is
#: a no-op when the editor does not pair, so treating every quote the same is
#: safe (see the module docstring's invariant).
QUOTES = "\"'`"

#: Trailing characters after which an editor indents the next line one level
#: deeper. Monaco/VS Code and CodeMirror both key off these.
_INDENTERS = ("{", "(", "[", ":")

#: A single step in a keystroke plan. One of:
#:   ("type",  "literal chunk")      -- type these at normal human cadence
#:   ("fast",  "    ")               -- type these as a burst (indent runs only)
#:   ("key",   "enter")              -- tap one named key
#:   ("chord", ("shift", "home"))    -- press a key combination together
Step = Tuple[str, object]

#: Languages where leading whitespace changes what the program means, so the
#: editor's guess can never be trusted. Matched against the caller's hint.
WHITESPACE_LANGUAGES = {
    "python", "py", "python2", "python3", "yaml", "yml", "haskell", "hs",
    "nim", "coffee", "coffeescript", "fsharp", "f#", "sass", "pug", "jade",
    "elm", "occam",
}

#: A block header in an indentation-structured language: a `def`/`if`/`for`...
#: line ending in a colon with nothing after it. Used only when the caller gave
#: no language hint. `match`/`case` are deliberately absent -- they would catch
#: every Java `switch` and push it onto the slow path for nothing.
_PY_BLOCK = re.compile(
    r"^[ \t]*(?:async[ \t]+def|def|class|if|elif|else|for|while|with|try|"
    r"except|finally)\b[^\n]*:[ \t]*(?:#.*)?$",
    re.MULTILINE,
)

#: A string literal that spans lines -- a Java text block, a Python triple-quote,
#: a JS template literal. Inside one, leading whitespace is *content*.
_MULTILINE_STRING = re.compile(r'("""|\'\'\'|`)[^\n]*\n.*?\1', re.DOTALL)


def indent_is_significant(source: str, language: Optional[str] = None) -> bool:
    """
    Whether this snippet's leading whitespace carries meaning.

    True means "the editor's auto-indent can never be trusted" -- use the exact,
    clear-and-re-type strategy. False means indentation is decoration and we can
    ride the editor's own auto-indent (see the module docstring).

    Order matters. A multi-line string literal makes whitespace significant in
    *any* language, including Java, so that check runs before the language hint.
    Tab-indented source is treated as significant too: a tab's visual width
    depends on the editor's tab size, so the column arithmetic this would need is
    not something we can do blind.
    """
    text = (source or "").replace("\r\n", "\n").replace("\r", "\n")
    if _MULTILINE_STRING.search(text):
        return True
    if any(_split_indent(line)[0].find("\t") >= 0 for line in text.split("\n")):
        return True
    lang = (language or "").strip().lower().lstrip(".")
    if lang:
        return lang in WHITESPACE_LANGUAGES
    return bool(_PY_BLOCK.search(text))


def _split_indent(line: str) -> Tuple[str, str]:
    """Split a line into (leading whitespace, the rest)."""
    stripped = line.lstrip(" \t")
    return line[: len(line) - len(stripped)], stripped


def _opens_block(body: str) -> bool:
    """Whether an editor will indent the line after this one."""
    return body.rstrip().endswith(_INDENTERS)


def infer_indent_unit(lines: List[str], default: int = 4) -> int:
    """
    How many spaces one indent level is in this snippet.

    The smallest step between two indentation levels actually used. Needed
    because the prediction in `editor` mode has to agree with the editor about
    what "one level deeper" means.
    """
    widths = sorted({len(_split_indent(l)[0]) for l in lines if l.strip()})
    steps = [b - a for a, b in zip(widths, widths[1:]) if b > a]
    if not steps:
        return default
    return max(1, min(8, min(steps)))


def plan_code_keystrokes(
    source: str,
    *,
    language: Optional[str] = None,
    indent_mode: str = "auto",
    indent_unit: Optional[int] = None,
    suppress_autoclose: bool = True,
    clear_autoindent: bool = True,
) -> List[Step]:
    """
    Turn source code into a keystroke plan that survives a code editor.

    Assumes the caller has put the cursor at end-of-document (empty or cleared
    editor); `type_code(..., replace=True)` guarantees that. See the module
    docstring for why that invariant matters.

    indent_mode:
      "auto"   -- `exact` when `indent_is_significant`, else `editor` (default)
      "editor" -- ride the editor's auto-indent, type only the shortfall
      "exact"  -- clear the auto-indent and re-type the indentation verbatim
      "raw"    -- do nothing about indentation at all (plain text boxes)

    `clear_autoindent=False` is kept as the old spelling of `indent_mode="raw"`.
    The other flags exist so the behaviour can be dialled per editor / per
    situation; the defaults are the robust setting.
    """
    text = (source or "").replace("\r\n", "\n").replace("\r", "\n")
    # Drop a single trailing newline so we do not leave a dangling blank line.
    if text.endswith("\n"):
        text = text[:-1]
    lines = text.split("\n")

    if not clear_autoindent:
        mode = "raw"
    elif indent_mode == "auto":
        mode = "exact" if indent_is_significant(text, language) else "editor"
    else:
        mode = indent_mode
    if mode not in ("editor", "exact", "raw"):
        raise ValueError(f"unknown indent_mode: {indent_mode!r}")

    unit = indent_unit or infer_indent_unit(lines)

    steps: List[Step] = []
    # The indent and content of the last non-blank line we typed. Blank lines are
    # skipped here on purpose: an editor carries the previous real line's indent
    # across a blank one, so resetting to zero would mispredict every line after
    # a paragraph break.
    prev_indent, prev_body = 0, ""

    for i, line in enumerate(lines):
        indent, body = _split_indent(line)

        if i > 0:
            steps.append(("key", "enter"))
            if mode == "exact":
                # Select whatever the editor auto-indented and delete it, so we
                # control the indentation exactly and start with no selection.
                steps.append(("chord", ("shift", "home")))
                steps.append(("key", "delete"))

        if mode == "editor":
            # Type only what the editor did NOT already put there. On a blank
            # line there is nothing to align, so leave the caret where it is.
            predicted = 0 if i == 0 else prev_indent + (unit if _opens_block(prev_body) else 0)
            shortfall = (len(indent) - predicted) if body else 0
            if shortfall > 0:
                steps.append(("fast", " " * shortfall))
        elif mode == "exact":
            if indent:
                steps.append(("fast", indent))
        else:                                        # raw
            if indent:
                steps.append(("type", indent))

        # Type the line, breaking after each opener/quote to delete its auto-pair.
        buf: List[str] = []
        for ch in body:
            buf.append(ch)
            if suppress_autoclose and (ch in OPENERS or ch in QUOTES):
                steps.append(("type", "".join(buf)))
                buf = []
                steps.append(("key", "delete"))     # remove the auto-inserted closer
        if buf:
            steps.append(("type", "".join(buf)))

        if body:
            prev_indent, prev_body = len(indent), body

    return _coalesce(steps)


def _coalesce(steps: List[Step]) -> List[Step]:
    """Merge adjacent literal chunks of the SAME cadence (keeps the plan minimal)."""
    out: List[Step] = []
    for kind, payload in steps:
        if kind in ("type", "fast") and out and out[-1][0] == kind:
            out[-1] = (kind, out[-1][1] + payload)  # type: ignore[operator]
        else:
            out.append((kind, payload))
    return out


def plan_char_count(steps: List[Step]) -> int:
    """How many literal characters the plan types (for reporting)."""
    return sum(len(p) for k, p in steps if k in ("type", "fast"))  # type: ignore[arg-type]


def run_plan(hands, steps: List[Step]) -> None:
    """
    Execute a keystroke plan through the human-timed hands layer.

    Literal chunks go through the same `type` path as everything else (human
    per-key jitter preserved); `fast` chunks are indentation runs, sent at the
    burst cadence a person produces holding a key or hitting Tab. Keys and chords
    go through `hotkey`.
    """
    for kind, payload in steps:
        if kind in ("type", "fast"):
            if payload:
                action = {"type": "type", "text": payload}
                if kind == "fast":
                    action["speed"] = "fast"
                hands.execute_action(action)
        elif kind == "key":
            hands.execute_action({"type": "hotkey", "keys": [payload]})
        elif kind == "chord":
            hands.execute_action({"type": "hotkey", "keys": list(payload)})  # type: ignore[arg-type]
        else:  # pragma: no cover - defensive
            raise ValueError(f"unknown keystroke step kind: {kind!r}")


def type_code(
    hands,
    source: str,
    *,
    replace: bool = True,
    language: Optional[str] = None,
    indent_mode: str = "auto",
    suppress_autoclose: bool = True,
    clear_autoindent: bool = True,
) -> int:
    """
    Type `source` into the focused code editor, neutralizing its quirks.

    replace: select-all and clear the editor first. This both replaces template
    /boilerplate code (the usual case) and establishes the
    end-of-document invariant the quirk-suppression relies on. Set False only if
    the editor is already empty and focused.

    language: the editor's language, when the caller knows it ("java", "python",
    ...). It only decides the indent strategy -- see `indent_is_significant`.

    Returns the number of literal code characters typed.
    """
    if replace:
        # Clear the editor: select everything, then delete the selection. Done as
        # two steps (not "type over the selection") so a first character that is
        # a bracket/quote cannot trigger "surround selection".
        hands.execute_action({"type": "hotkey", "keys": ["ctrl", "a"]})
        hands.execute_action({"type": "hotkey", "keys": ["delete"]})

    steps = plan_code_keystrokes(
        source,
        language=language,
        indent_mode=indent_mode,
        suppress_autoclose=suppress_autoclose,
        clear_autoindent=clear_autoindent,
    )
    run_plan(hands, steps)
    return plan_char_count(steps)
