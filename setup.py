from setuptools import setup

setup(
    name='claim_tagger',
    version='0.1',
    packages=['claim_tagger', 'claim_tagger.ml', 'claim_tagger.test', 'claim_tagger.config',
              'claim_tagger.featurizer'],
    url='',
    license='',
    author='Danilo S. Carvalho',
    author_email='danilo@jaist.ac.jp',
    description='TDV feature-based LSTM tagger for patent claims.',
    install_requires=[
        'saf',
        'wikt_morphodecomp',
        'numpy',
        'keras'
    ]
)
