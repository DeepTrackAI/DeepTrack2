import setuptools
import pkg_resources

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    required = fh.read().splitlines()

# Remove sphinx from requirements
required = [x for x in required if not x.startswith("Sphinx")]
required = [x for x in required if not x.startswith("pydata-sphinx-theme")]


installed = [pkg.key for pkg in pkg_resources.working_set]
if (
    "tensorflow" not in installed
    or pkg_resources.working_set.by_key["tensorflow"].version[0] == "2"
):
    required.append("tensorflow_addons")


setuptools.setup(
    name="deeptrack",  # Replace with your own username
    version="1.6.0",
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
    python_requires=">=3.6",
)
