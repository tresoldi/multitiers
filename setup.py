import pathlib
from setuptools import setup

# The directory containing this file
LOCAL_PATH = pathlib.Path(__file__).parent

# The text of the README file
README_FILE = (LOCAL_PATH / "README.md").read_text()

# Load requirements, so they are listed in a single place
with open("requirements.txt") as fp:
    install_requires = [dep.strip() for dep in fp.readlines()]

# This call to setup() does all the work
setup(
    author_email="tresoldi@shh.mpg.de",
    author="Tiago Tresoldi",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Software Development :: Libraries",
    ],
    description="A library for multi-tiered sequence representation of linguistic data.",
    entry_points={"console_scripts": ["multitiers=multitiers.__main__:main"]},
    include_package_data=True,
    install_requires=install_requires,
    keywords=["tiers", "multitiers", "phonology", "historical linguistics"],
    license="MIT",
    long_description_content_type="text/markdown",
    long_description=README_FILE,
    name="multitiers",
    packages=["multitiers", "clts-master"],
    project_urls={"Documentation": "https://multitiers.readthedocs.io"},
    test_suite="tests",
    tests_require=[],
    url="https://github.com/tresoldi/multitiers",
    version="0.1",
    zip_safe=False,
)
