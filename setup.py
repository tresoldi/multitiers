import pathlib
from setuptools import setup

# The directory containing this file
LOCAL_PATH = pathlib.Path(__file__).parent

# The text of the README file
README_FILE = (LOCAL_PATH / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="multitiers",
    version="0.1",
    description="A library for multi-tiered sequence representation of linguistic data.",
    long_description=README_FILE,
    long_description_content_type="text/markdown",
    url="https://github.com/tresoldi/multitiers",
    project_urls={"Documentation": "https://multitiers.readthedocs.io"},
    author="Tiago Tresoldi",
    author_email="tresoldi@shh.mpg.de",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries",
    ],
    packages=["multitiers", "clts-master"],
    keywords=["tiers", "multitiers", "phonology", "historical linguistics"],
    include_package_data=True,
    install_requires=["pyclts"],
    entry_points={"console_scripts": ["multitiers=multitiers.__main__:main"]},
    test_suite="tests",
    tests_require=[],
    zip_safe=False,
)
