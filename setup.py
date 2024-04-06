"""Python setup.py for dstool package"""
import io
import os
from setuptools import find_packages, setup


def read(*paths, **kwargs):
    """Read the contents of a text file safely.
    >>> read("dstool", "VERSION")
    '0.1.0'
    >>> read("README.md")
    ...
    """

    content = ""
    with io.open(
        os.path.join(os.path.dirname(__file__), *paths),
        encoding=kwargs.get("encoding", "utf8"),
    ) as open_file:
        content = open_file.read().strip()
    return content


def read_requirements(path):
    return [
        line.strip()
        for line in read(path).split("\n")
        if not line.startswith(('"', "#", "-", "git+"))
    ]


setup(
    name="dstool",
    version=read("dstool", "VERSION"),
    description="Awesome dstool created by xzyaoi",
    url="https://github.com/xzyaoi/dstool/",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    author="xzyaoi",
    packages=find_packages(exclude=["tests", ".github"]),
    install_requires=read_requirements("requirements.txt"),
    entry_points={
        "dstool": ["dstool = dstool.__main__:main"]
    },
    extras_require={"test": read_requirements("requirements-test.txt")},
)
