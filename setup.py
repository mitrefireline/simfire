import setuptools
import os

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

if os.environ.get('CI_COMMIT_TAG'):
    version = os.environ['CI_COMMIT_TAG']
else:
    version = '0.0.0'

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name='rothsim',
    version=version,
    install_requires=requirements,
    description='Rothermel fire modeler that uses PyGame for display',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://gitlab.mitre.org/fireline/rothermel-modeling',
    project_urls={
        'Documentation': 'https://fireline.pages.mitre.org/rothermel-modeling/'
    },
    classifiers=['Programming Language :: Python :: 3'],
    package_dir={'': 'src'},
    packages=setuptools.find_packages(where='src'),
    author='The MITRE Corporation',
    author_email='twelsh@mitre.org',
)
