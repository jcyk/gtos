#!/usr/bin/env python

import sys
import os
try:
    from setuptools import setup, find_packages
except ImportError:
    from disutils.core import setup


here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "README.md")) as f:
    README = f.read()


setup(name="smatch",
      version="1.0",
      description="Smatch (semantic match) tool",
      long_description=README,
      author="Shu Cai",
      author_email="shucai.work@gmail.com",
      url="https://github.com/snowblink14/smatch",
      license="MIT",
      py_modules=["smatch", "amr"],
      scripts=["smatch.py"],
      )

