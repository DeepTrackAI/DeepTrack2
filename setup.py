import setuptools
import pkg_resources

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    required = fh.read().splitlines()

# Remove sphinx from requirements
required = [x for x in required if not x.startswith("Sphinx")]
required = [x for x in required if not x.startswith("pydata-sphinx-theme")]




setuptools.setup(
    name="deeptrack",  # Replace with your own username
    version="1.5.2",
    author="Benjamin Midtvedt",
    author_email="benjamin.midtvedt@physics.gu.se",
    description="A deep learning oriented microscopy image simulation package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/softmatterlab/DeepTrack-2.0/",
    install_requires=required,
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    extras_requires={"tensorflow": ["tensorflow<=2.10", "tensorflow-probability", "tensorflow-datasets", "tensorflow_addons"]},
    python_requires=">=3.6",
)
