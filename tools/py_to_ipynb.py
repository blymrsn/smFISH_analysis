"""Convert the percent-format notebooks (notebooks/*.py, '# %%' cells) to .ipynb.

    python tools/py_to_ipynb.py              # all notebooks/[0-9]*.py
(jupytext --to ipynb does the same if installed.)
"""
import glob
import re
import sys

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

for f in sys.argv[1:] or sorted(glob.glob("notebooks/[0-9]*.py")):
    parts = re.split(r"^# %%", open(f).read(), flags=re.M)[1:]
    cells = []
    for p in parts:
        header, _, body = p.partition("\n")
        if "[markdown]" in header:
            cells.append(new_markdown_cell("\n".join(l[2:] if l.startswith("# ") else l.lstrip("#")
                                                     for l in body.strip("\n").splitlines())))
        else:
            cells.append(new_code_cell(body.strip("\n")))
    nb = new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3",
                                                            "language": "python"}})
    nbformat.validate(nb)
    nbformat.write(nb, f[:-3] + ".ipynb")
    print("wrote", f[:-3] + ".ipynb")
