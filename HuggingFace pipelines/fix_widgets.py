import nbformat
from pathlib import Path

# Path to the notebook
nb_path = Path("/Users/vahedi/Library/CloudStorage/OneDrive-RoskildeUniversitet/Python Projects/LLMs/HuggingFace pipelines/PipelineAPIs.ipynb")

# Read as v4
nb = nbformat.read(nb_path, as_version=4)

# If metadata.widgets exists but has no "state", add an empty one
if "widgets" in nb.metadata and "state" not in nb.metadata.widgets:
    print("Adding missing 'state' to metadata.widgets…")
    nb.metadata.widgets["state"] = {}

# Write it back
nbformat.write(nb, nb_path)
print("Done.")
