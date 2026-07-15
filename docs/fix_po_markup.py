"""Fix reST markup and spacing issues in translated .po files.

This script post-processes translated ``.po`` files to fix three classes of
problems in the rendered Japanese documentation:

1. Inline markup adjacency: docutils only recognizes inline markup (e.g.
   ``...``, :func:`...`) when it is delimited by whitespace or certain
   punctuation.  Japanese translations often place markup directly next to
   Japanese characters (e.g. この関数は``-mmax(-x)``として), which makes
   docutils render the raw backticks.  This script inserts spaces around
   inline markup where needed.

2. Flattened composite messages: Sphinx's gettext builder extracts a field
   body that contains a nested bullet list (e.g. a "Returns:" section
   describing a named tuple) twice: once as a single flattened paragraph and
   once per structural element.  If the flattened message is translated, its
   translation replaces the whole structure and the list renders as flat
   text.  This script empties such translations so that Sphinx falls back to
   the per-element translations, preserving the list structure.

3. Japanese/Latin spacing: following common Japanese technical writing
   style, a half-width space is inserted between Japanese characters and
   half-width alphanumerics (e.g. NaNになります -> NaN になります), and
   spaces adjacent to Japanese punctuation (、。（）etc.) are removed.
   Inline markup that would touch punctuation is delimited with an escaped
   space ("\\ ") instead, which renders without a visible space.  In
   doctest blocks, only the ``#`` comment parts are adjusted.
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

# Japanese characters that participate in spacing: kana, iteration marks,
# and CJK ideographs.  Punctuation (、。（）：etc.) is deliberately excluded
# so that no space is inserted next to it.
JA_CHAR = "[々〆ぁ-ゟ゠-ヿ" "㐀-䶿一-鿿豈-﫿]"
# Half-width characters that should be separated from Japanese characters:
# alphanumerics, Greek letters and the plus-minus sign.
HW_CHAR = "[0-9A-Za-z±Ͱ-Ͽ]"
JA_SPACING_RES = [
    # Japanese character followed by a half-width character (optionally a
    # signed number, e.g. デフォルトは-1).
    re.compile(f"({JA_CHAR})([-+±]?{HW_CHAR})"),
    # Half-width character followed by a Japanese character.
    re.compile(f"({HW_CHAR})({JA_CHAR})"),
]
# Japanese punctuation must not be surrounded by spaces.
JA_PUNCT = "、。（）「」『』：；！？・"
JA_PUNCT_SPACE_RES = [
    re.compile(f"([{JA_PUNCT}]) +"),
    re.compile(f" +([{JA_PUNCT}])"),
]
# Characters that take a regular (visible) space when adjacent to inline
# markup; any other character takes an escaped space ("\\ "), which is
# removed from the rendered output.
WORD_CHAR_RE = re.compile(f"{JA_CHAR}|{HW_CHAR}")

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


def _markup_separator(ch: str) -> str:
    """Return the separator to insert between inline markup and ``ch``.

    Word-like characters get a regular space (visible in the output);
    punctuation and other symbols get an escaped space ("\\ "), which
    delimits the markup without rendering a space.
    """
    return " " if WORD_CHAR_RE.fullmatch(ch) else "\\ "


def fix_inline_markup_adjacency(text: str) -> str:
    """Insert spaces around inline markup adjacent to CJK text."""
    insertions: List[Tuple[int, str]] = []
    for match in INLINE_MARKUP_RE.finditer(text):
        start, end = match.span()
        if start > 0 and not _may_precede_markup(text[start - 1]):
            insertions.append((start, _markup_separator(text[start - 1])))
        if end < len(text) and not _may_follow_markup(text[end]):
            insertions.append((end, _markup_separator(text[end])))
    for pos, separator in sorted(insertions, reverse=True):
        text = text[:pos] + separator + text[pos:]
    return text


def _space_ja_text(text: str) -> str:
    """Insert spaces between Japanese characters and alphanumerics."""
    for pattern in JA_PUNCT_SPACE_RES:
        text = pattern.sub(r"\1", text)
    for pattern in JA_SPACING_RES:
        text = pattern.sub(r"\1 \2", text)
    return text


def add_ja_spacing(text: str) -> str:
    """Apply Japanese/Latin spacing outside inline markup spans."""
    parts: List[str] = []
    last = 0
    for match in INLINE_MARKUP_RE.finditer(text):
        parts.append(_space_ja_text(text[last : match.start()]))
        parts.append(match.group(0))
        last = match.end()
    parts.append(_space_ja_text(text[last:]))
    return "".join(parts)


def space_code_block_comments(text: str) -> str:
    """Apply Japanese/Latin spacing to ``#`` comments in doctest blocks."""
    lines = text.split("\n")
    for i, line in enumerate(lines):
        match = re.match(r"^(\s*(?:>>>|\.\.\.)\s[^#]*)(#.*)$", line)
        if match:
            lines[i] = match.group(1) + _space_ja_text(match.group(2))
    return "\n".join(lines)


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
        # Code blocks are rendered literally; only adjust their comments.
        return space_code_block_comments(msgstr)
    if "\\`" in msgstr:
        # Escaped backticks (e.g. in :sphinx_autodoc_typehints_type: roles)
        # are beyond what INLINE_MARKUP_RE can parse reliably.
        return msgstr
    # Escaped spaces ("\ ") inserted by earlier runs render as no space at
    # all; replace them with regular spaces to match the spacing style.
    msgstr = msgstr.replace("\\ ", " ")
    msgstr = add_ja_spacing(msgstr)
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
