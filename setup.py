#!/usr/bin/env python

from distutils.core import setup

with open('requirements.txt') as fp:
    requirements = fp.read().splitlines()

setup(name='deep-belief-network',
      version='0.7.0',
      description='Python implementation of Deep Belief Networks',
      packages=['dbn'],
      install_requires=requirements,
      )
