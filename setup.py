from setuptools import setup

setup(
    name='logistic_dml',
    version='1.0.0',
    description='Python implementation of Logistic Double ML',
    packages=['logistic_dml'],
    install_requires=[
	'pandas>=1.5.2',
	'scipy>=1.9.3',
	'numpy>=1.23.5',
    'scikit-learn>=1.3.0',
    ],
    author='damerdji',
    author_email='salim.damerdji@stats.ox.ac.uk',
    url='https://github.com/sdamerdji/logistic_dml',
)
