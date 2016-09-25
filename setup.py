#!/usr/bin/env python

from distutils.core import setup

setup(name='deep-belief-network',
      version='0.1.7',
      description='Python implementation of Deep Belief Networks',
      packages=['dbn'],
      install_requires=['numpy>=1.9.2', 'scipy', 'scikit-learn>=0.16.1']
      )
