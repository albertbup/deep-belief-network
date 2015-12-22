#!/usr/bin/env python

from distutils.core import setup

setup(name='deep-belief-network',
      version='0.1.1',
      description='Python implementation of Deep Belief Networks',
      packages=['dbn'],
      install_requires=['scikit-learn>=0.16.1', 'numpy>=1.9.2']
      )