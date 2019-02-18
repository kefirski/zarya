from os.path import join, dirname

from setuptools import setup, find_packages

from zarya import __version__

setup(
    name="zarya",
    version=__version__,
    packages=find_packages(),
    long_description=open(join(dirname(__file__), "README.md")).read(),
)
