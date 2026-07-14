"""Fix reST markup issues in translated .po files.

This script post-processes translated ``.po`` files to fix two classes of
problems that break the rendered Japanese documentation:

1. Inline markup adjacency: docutils only recognizes inline markup (e.g.
   ``...``, :func:`...`) when it is delimited by whitespace or certain
   punctuation.  Japanese translations often place markup directly next to
   Japanese characters (e.g. この関数は``-mmax(-x)``として), which makes
   docutils render the raw backticks.  This script inserts escaped spaces
   ("\\ ") around inline markup where needed; escaped whitespace is removed
   from the rendered output.

2. Flattened composite messages: Sphinx's gettext builder extracts a field
   body that contains a nested bullet list (e.g. a "Returns:" section
   describing a named tuple) twice: once as a single flattened paragraph and
   once per structural element.  If the flattened message is translated, its
   translation replaces the whole structure and the list renders as flat
   text.  This script empties such translations so that Sphinx falls back to
   the per-element translations, preserving the list structure.
"""

import argparse
import logging
import re
import unicodedata
from pathlib import Path
from typing import List
from typing import Tuple

logger = logging.getLogger(__name__)

# Matches flattened field bodies such as
# "A named tuple of ``(values, indices)``:      - ``values`` (Tensor): ..."
# (an introductory sentence ending with a colon, followed by run-on spaces
# and bullet items).  Regular prose never contains ":<spaces>- ".
FLATTENED_COMPOSITE_RE = re.compile(r":\s{2,}-\s")

# Inline markup spans handled by this script.  Roles and literals are
# matched before single-backtick title references.
INLINE_MARKUP_RE = re.compile(
    r":[A-Za-z0-9_.+:-]+:`[^`\n]+`"  # role, e.g. :func:`mmax`
    r"|``[^`\n]+``"  # inline literal, e.g. ``span - 1``
    r"|`[^`\n]+`_{0,2}"  # title reference or link, e.g. `dim` / `x`_
)

# docutils inline markup recognition rules:
# https://docutils.sourceforge.io/docs/ref/rst/restructuredtext.html#inline-markup-recognition-rules
OPENERS = set("-:/'\"<([{")
OPENER_CATEGORIES = {"Ps", "Pi", "Pf", "Pd", "Po"}
CLOSERS = set("-.,:;!?\\/'\")]}>")
CLOSER_CATEGORIES = {"Pe", "Pi", "Pf", "Pd", "Po"}


def _may_precede_markup(ch: str) -> bool:
    return (
        ch.isspace()
        or ch in OPENERS
        or unicodedata.category(ch) in OPENER_CATEGORIES
    )


def _may_follow_markup(ch: str) -> bool:
    return (
        ch.isspace()
        or ch in CLOSERS
        or unicodedata.category(ch) in CLOSER_CATEGORIES
    )


def fix_inline_markup_adjacency(text: str) -> str:
    """Insert escaped spaces around inline markup adjacent to CJK text."""
    insertions: List[int] = []
    for match in INLINE_MARKUP_RE.finditer(text):
        start, end = match.span()
        if start > 0 and not _may_precede_markup(text[start - 1]):
            insertions.append(start)
        if end < len(text) and not _may_follow_markup(text[end]):
            insertions.append(end)
    for pos in sorted(insertions, reverse=True):
        text = text[:pos] + "\\ " + text[pos:]
    return text


def _is_code_block(msgid: str) -> bool:
    return msgid.lstrip().startswith(">>>")


def fix_msgstr(msgid: str, msgstr: str) -> str:
    """Return the fixed msgstr for the given msgid/msgstr pair."""
    if not msgid or not msgstr:
        return msgstr
    if FLATTENED_COMPOSITE_RE.search(msgid):
        # Empty the translation of a flattened composite message so that
        # Sphinx uses the structured per-element translations instead.
        return ""
    if msgstr == msgid:
        # An identical translation renders exactly like the original, which
        # is well-formed by construction.
        return msgstr
    if _is_code_block(msgid) or _is_code_block(msgstr):
        # Code blocks are rendered literally; escaped spaces would corrupt
        # the code.
        return msgstr
    if "\\`" in msgstr:
        # Escaped backticks (e.g. in :sphinx_autodoc_typehints_type: roles)
        # are beyond what INLINE_MARKUP_RE can parse reliably.
        return msgstr
    return fix_inline_markup_adjacency(msgstr)


def decode_po_string(lines: List[str], keyword: str) -> str:
    """Decode a (possibly multi-line) PO string into raw text."""
    text = ""
    for line in lines:
        line = line.strip()
        if line.startswith(keyword):
            line = line[len(keyword) :].strip()
        assert line.startswith('"') and line.endswith('"'), line
        text += line[1:-1]
    return (
        text.replace("\\\\", "\x00")
        .replace('\\"', '"')
        .replace("\\n", "\n")
        .replace("\\t", "\t")
        .replace("\x00", "\\")
    )


def encode_po_string(text: str) -> str:
    """Encode raw text into a single-line PO string body."""
    return (
        text.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\t", "\\t")
    )


def process_po_file(file_path: Path) -> int:
    """Fix all msgstr entries in a .po file.  Returns the number of fixes."""
    lines = file_path.read_text(encoding="utf-8").splitlines(keepends=True)
    output: List[str] = []
    fixes = 0
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.startswith("msgid "):
            output.append(line)
            i += 1
            continue

        msgid_lines = [line]
        i += 1
        while i < len(lines) and lines[i].startswith('"'):
            msgid_lines.append(lines[i])
            i += 1
        output.extend(msgid_lines)

        if i >= len(lines) or not lines[i].startswith("msgstr "):
            continue
        msgstr_lines = [lines[i]]
        i += 1
        while i < len(lines) and lines[i].startswith('"'):
            msgstr_lines.append(lines[i])
            i += 1

        msgid = decode_po_string(msgid_lines, "msgid")
        msgstr = decode_po_string(msgstr_lines, "msgstr")
        fixed = fix_msgstr(msgid, msgstr)
        if fixed == msgstr:
            output.extend(msgstr_lines)
        else:
            output.append(f'msgstr "{encode_po_string(fixed)}"\n')
            fixes += 1

    if fixes:
        file_path.write_text("".join(output), encoding="utf-8")
    return fixes


def find_po_files(to_lang: str) -> List[Path]:
    base = Path(__file__).parent / "locale" / to_lang / "LC_MESSAGES"
    return sorted(base.glob("**/*.po")) if base.exists() else []


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fix reST markup issues in translated .po files"
    )
    parser.add_argument(
        "--to-lang", default="ja", help="Target language (default: ja)"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    po_files = find_po_files(args.to_lang)
    if not po_files:
        logger.warning("No .po files found for language: %s", args.to_lang)
        return

    total = 0
    for po_file in po_files:
        fixes = process_po_file(po_file)
        if fixes:
            logger.info("%s: fixed %d entries", po_file, fixes)
        total += fixes
    logger.info("Fixed %d entries in total", total)


if __name__ == "__main__":
    main()
