from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    required = fh.read().splitlines()

# Remove sphinx from requirements
required = [x for x in required if not x.startswith("Sphinx")]
required = [x for x in required if not x.startswith("pydata-sphinx-theme")]

setup(
    name="deeptrack",
    version="2.0.0",
    license="MIT",
    packages=find_packages(),
    author=(
        "Benjamin Midtvedt, Jesus Pineda, Henrik Klein Moberg, "
        "Harshith Bachimanchi, Carlo Manzo, Giovanni Volpe"
    ),
    description=(
        "A deep learning framework to enhance microscopy, "
        "developed by DeepTrackAI."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DeepTrackAI/DeepTrack2",
    install_requires=required,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    extras_requires={"tensorflow": ["tensorflow<=2.10", "tensorflow-probability", "tensorflow-datasets", "tensorflow_addons"]},
    python_requires=">=3.8",
)
