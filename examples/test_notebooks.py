import glob
import os


def test_notebooks():
    """
    Test all notebooks in the examples directory.
    """

    notebooks = glob.glob(
        os.path.join("examples", "tutorials", "*.ipynb"), recursive=True
    )
    failed_runs = []
    for notebook in notebooks:
        print(f"Testing notebook: {notebook}")
        # Allow errors to be raised.
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
