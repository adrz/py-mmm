#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(name='py-mmm',
      version='0.1',
      description='Implementation of EM for mixture markov model',
      url='http://github.com/adrz/py-mmm',
      author='Adrien N.',
      author_email='adrien.nouvellet@gmail.com',
      license='MIT',
      packages=find_packages(exclude=('tests', 'tests.*')),
      install_requires=["matplotlib", "numpy"],
      zip_safe=False)
