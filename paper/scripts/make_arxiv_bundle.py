"""Write paper/arxiv_upload.tar.gz, the source bundle to upload to arXiv.

Usage, after building the paper (the bundle needs sotaku_short.bbl): python -m paper.scripts.make_arxiv_bundle

The bundle holds the three .tex files, the built bibliography, the generated tables and only the
figures the paper includes. arXiv does not run bibtex, so the .bbl file must be the current one.
"""

import re
import tarfile
from pathlib import Path

PAPER = Path(__file__).resolve().parent.parent
SOURCES = ("sotaku_short.tex", "body.tex", "appendix.tex")


def bundle_files():
    figures = set()
    for source in SOURCES:
        figures |= set(re.findall(r"includegraphics[^{]*\{([^}]*)\}", (PAPER / source).read_text()))
    files = [*SOURCES, "sotaku_short.bbl", *sorted(str(path.relative_to(PAPER)) for path in (PAPER / "generated").glob("*.tex"))]
    for name in sorted(figures):
        extension = "pdf" if (PAPER / "figures" / f"{name}.pdf").exists() else "png"
        files.append(f"figures/{name}.{extension}")
    return files


def main():
    files = bundle_files()
    target = PAPER / "arxiv_upload.tar.gz"
    with tarfile.open(target, "w:gz") as bundle:
        for name in files:
            bundle.add(PAPER / name, arcname=name)
    print(f"Wrote {target} with {len(files)} files")


if __name__ == "__main__":
    main()
