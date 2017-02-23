#!/usr/bin/env python

from distutils.core import setup

setup(name='deep-belief-network',
      version='0.4.0',
      description='Python implementation of Deep Belief Networks',
      packages=['dbn'],
      install_requires=['numpy==1.12.0',
                        'scipy==0.18.1',
                        'scikit-learn==0.18.1',
                        'tensorflow==1.0.0'
                        ]
      )
