#!/usr/bin/env python3
"""Post-process sphinx-apidoc output to add function links."""

import re
from pathlib import Path

def process_functions_rst():
    """Add autosummary to functions package file."""
    functions_rst = Path("api/qfeval_functions.functions.rst")
    if not functions_rst.exists():
        return
    
    content = functions_rst.read_text()
    
    # Find the automodule section
    automodule_pattern = r'(.. automodule:: qfeval_functions\.functions\n(?:   .*\n)*)'
    match = re.search(automodule_pattern, content)
    
    if not match:
        return
    
    # Get all public functions exported from the package so that functions
    # sharing a module with another one (e.g. pca_cov, soft_topk,
    # reduce_nan_patterns) also get their own pages.
    import inspect
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import qfeval_functions.functions

    functions = sorted(
        name
        for name, obj in vars(qfeval_functions.functions).items()
        if inspect.isfunction(obj)
        and obj.__module__.startswith("qfeval_functions.functions.")
    )
    
    # Replace the automodule section with just the package description and autosummary
    # Also update the title to "Functions (qfeval_functions.functions)"
    title = "Functions (qfeval_functions.functions)"
    package_header = content[:match.start()]
    package_header = package_header.replace(
        "qfeval\\_functions.functions package\n===================================",
        title + "\n" + "=" * len(title)
    )
    
    # Create autosummary section with table format
    autosummary = "\n.. currentmodule:: qfeval_functions.functions\n\n.. autosummary::\n   :toctree: .\n   :nosignatures:\n\n"
    for func in functions:
        autosummary += f"   {func}\n"
    
    # Remove the Submodules section and toctree
    remaining_content = content[match.end():]
    # Remove everything from "Submodules" onwards
    submodules_idx = remaining_content.find("Submodules")
    if submodules_idx != -1:
        remaining_content = remaining_content[:submodules_idx].rstrip()
    
    # Build new content without automodule
    new_content = package_header + autosummary
    
    functions_rst.write_text(new_content)
    print("Updated qfeval_functions.functions.rst with autosummary")

def process_random_rst():
    """Rewrite the random package file to use autosummary."""
    random_rst = Path("api/qfeval_functions.random.rst")
    if not random_rst.exists():
        return

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import qfeval_functions.random

    title = "Random (qfeval_functions.random)"
    content = title + "\n" + "=" * len(title) + "\n"
    content += "\n.. currentmodule:: qfeval_functions.random\n"
    content += "\n.. autosummary::\n   :toctree: .\n   :nosignatures:\n\n"
    for name in qfeval_functions.random.__all__:
        content += f"   {name}\n"

    random_rst.write_text(content)
    print("Updated qfeval_functions.random.rst with autosummary")


def process_main_rst():
    """Update main qfeval_functions.rst to use autosummary for functions."""
    main_rst = Path("api/qfeval_functions.rst")
    if not main_rst.exists():
        return
    
    content = main_rst.read_text()
    
    # Find the toctree for functions
    functions_pattern = r'(.. toctree::\n   :maxdepth: 4\n\n   qfeval_functions\.functions)'
    match = re.search(functions_pattern, content)
    
    if not match:
        return
    
    # Replace with direct link to functions (no toctree)
    new_content = content.replace(
        ".. toctree::\n   :maxdepth: 4\n\n   qfeval_functions.functions",
        "* :doc:`Functions <qfeval_functions.functions>`"
    )
    
    main_rst.write_text(new_content)
    print("Updated qfeval_functions.rst")

if __name__ == "__main__":
    process_functions_rst()
    process_random_rst()
    process_main_rst()