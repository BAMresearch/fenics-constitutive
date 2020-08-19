from glob import glob
from pathlib import Path


sources = glob("examples/*.py") + glob("examples/*.ipynb")
targets = [str(Path(s).with_suffix(".rst")) for s in sources]

def task_website():
    return {
            "file_dep": targets + ["conf.py", "index.rst"],
            "actions": ["sphinx-build . website"],
            "verbosity": 2
            }


def task_convert():
    for source, target in zip(sources, targets):
        if source.endswith(".py"):
            yield {
                    "basename": "convert via to_rst.py",
                    "name": source,
                    "targets": [target],
                    "file_dep" : [source, "to_rst.py"],
                    "actions": [f"python3 to_rst.py {source} > {target}"]
                        }
        elif source.endswith(".ipynb"):
            yield   {
                "basename": "convert via jupyter",
                "name" : source,
                "targets": [target],
                "file_dep" : [source],
                "actions": [f"jupyter-nbconvert --to rst {source}"]
                    }
        else:
            raise RuntimeError(f"Unknown format of {source}. *.py, *.ipynb supported.")
            # yield
            # {}

