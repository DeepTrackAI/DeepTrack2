import re

def main():
    """Create README-pypi.md from README.md."""

    with open("README.md", "r", encoding="utf-8") as f:
        text = f.read()

    # Regex to remove content between special markers
    pattern = r"<!-- GH_ONLY_START -->.*?<!-- GH_ONLY_END -->"
    cleaned_text = re.sub(pattern, "", text, flags=re.DOTALL)

    with open("README-pypi.md", "w", encoding="utf-8") as f:
        f.write(cleaned_text)

    print("README-pypi.md has been generated successfully!")

if __name__ == "__main__":
    main()
