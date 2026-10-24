"""Build <team>_submission.zip in the layout required by the challenge."""
import os, sys, zipfile

team = sys.argv[1] if len(sys.argv) > 1 else "team"
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = os.path.join(root, f"{team}_submission.zip")
code = "code/business_entity_resolution"
files = {
    "output/matching_results.tsv": "output/matching_results.tsv",
    "output/candidate_pairs.tsv": "output/candidate_pairs.tsv",
    "Documentation_template.md": "Documentation_template.md",
    f"{code}/README.md": "SOLUTION_README.md",
    f"{code}/requirements.txt": "requirements.txt",
}
for f in sorted(os.listdir(os.path.join(root, "src"))):
    if f.endswith((".py", ".sh")):
        files[f"{code}/src/{f}"] = f"src/{f}"
for f in sorted(os.listdir(os.path.join(root, "tests"))):
    if f.endswith(".py"):
        files[f"{code}/tests/{f}"] = f"tests/{f}"
import glob
for f in sorted(glob.glob(os.path.join(root, "cloud_package*", "*"))):
    if f.endswith((".py", ".ipynb", ".txt")):  # GPU notebooks for the cross-encoders (data files are not shipped)
        rel = os.path.relpath(f, root).replace(os.sep, "/")
        files[f"{code}/{rel}"] = rel
for f in ("utils/validate_submission.py",):
    if os.path.exists(os.path.join(root, f)):
        files[f"{code}/{f}"] = f
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for arc, src in files.items():
        z.write(os.path.join(root, src), arc)
        print("added", arc)
print("wrote", out, round(os.path.getsize(out) / 1e6, 1), "MB")
