import glob
import os

# HOW TO:
# go to root and run pip install -e .
# Run python examples/test_notebooks.py


def test_notebooks():
    """
    Test all notebooks in the examples directory.
    """

    notebooks = glob.glob(os.path.join("examples", "**", "*.ipynb"), recursive=True)
    failed_runs = []
    for notebook in notebooks:
        print(f"Testing notebook: {notebook}...")

        # Allow errors to be raised.
        out = os.popen(f'git diff --name-only "{notebook}"').read()
        if out:
            print("Notebook already ran since last git commit... skipping")
            continue
        out = os.system(
            f'jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=None "{notebook}"'
        )

        # Check if notebook ran successfully.
        if out != 0:
            failed_runs.append(notebook)

    if failed_runs:
        print("Failed runs:")
        for notebook in failed_runs:
            print(f"\t{notebook}")
        raise ValueError("Some notebooks failed to run.")
    else:
        print("All notebooks ran successfully.")


if __name__ == "__main__":
    print("Testing notebooks...")
    test_notebooks()
