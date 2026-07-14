import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "qfeval-functions"
copyright = "2025, qfeval-functions contributors"
author = "qfeval-functions contributors"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    # Always show the full function lists of all packages in the sidebar.
    "collapse_navigation": False,
}

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
    "show-inheritance": True,
}

# Remove module path from function signatures
add_module_names = False

# Do not add function/class signature entries to the table of contents;
# each API page documents exactly one object, so the page title is enough.
toc_object_entries = False

# Generate separate pages for each module
autosummary_generate = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

# -- setting for intl ------
locale_dirs = ["locale/"]
gettext_compact = False
# Extract code examples (doctest/literal blocks) into .po files so that
# comments in examples can be translated.
gettext_additional_targets = ["literal-block", "doctest-block"]


def setup(app):  # type: ignore[no-untyped-def]
    from sphinx.transforms.post_transforms.code import (
        TrimDoctestFlagsTransform,
    )

    class EarlyTrimDoctestFlagsTransform(TrimDoctestFlagsTransform):
        """Trim doctest flags (e.g. <BLANKLINE>) before the Locale transform.

        The gettext builder extracts doctest blocks after trim_doctest_flags
        has been applied, but translation lookup happens on the untrimmed
        text.  Doctest blocks containing flags would therefore never match
        their catalog entries.  Running the trim before the Locale transform
        (priority 20) keeps both sides consistent.
        """

        default_priority = 15

    app.add_transform(EarlyTrimDoctestFlagsTransform)
