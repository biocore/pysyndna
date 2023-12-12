# ----------------------------------------------------------------------------
# Copyright (c) 2023, Amanda Birmingham.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------
from setuptools import setup, find_packages

setup(name='pysyndna',
      version="0.1.0",
      long_description="A python implementation and extension of SynDNA "
                       "('a Synthetic DNA Spike-in Method for Absolute "
                       "Quantification of Shotgun Metagenomic Sequencing', "
                       "Zaramela et al., mSystems 2022)",
      license='BSD-3-Clause',
      description='Python implementation of the SynDNA algorithm',
      author="Amanda Birmingham",
      author_email="abirmingham@ucsd.edu",
      url='https://github.com/AmandaBirmingham/pysyndna',
      packages=find_packages(),
      include_package_data=True,
      package_data={
          'pysyndna': [
              '*.*',
              'tests/data/*.*',
            ]},
      # making sure that numpy is installed before biom
      setup_requires=['numpy', 'cython'],
      install_requires=['pandas<2.0', 'scipy', 'scikit-learn', 'pyyaml',
                        'biom-format',  'nose', 'pep8', 'flake8'],
      )
