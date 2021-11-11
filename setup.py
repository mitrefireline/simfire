import os
import setuptools

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

if os.environ.get('CI_COMMIT_TAG'):
    version = os.environ['CI_COMMIT_TAG']
else:
    version = '0.0.0'

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

# Find the packages and correctly label name them for `setup`
# Make sure to exclude _tests
packages = setuptools.find_packages(where='src', exclude=['*_tests*'])
packages = ['rothsim.' + p for p in packages] + ['rothsim'] + ['assets']

setuptools.setup(
    name='rothsim',
    version=version,
    install_requires=requirements,
    description='Rothermel fire modeler using PyGame for display',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://gitlab.mitre.org/fireline/rothermel-modeling',
    project_urls={
        'Documentation': 'https://fireline.pages.mitre.org/rothermel-modeling/'
    },
    classifiers=['Programming Language :: Python :: 3'],
    package_dir={
        'rothsim': 'src',
        'assets': 'assets'
    },
    packages=packages,
    package_data={'assets': ['../assets/**/*']},
    author='The MITRE Corporation',
    author_email='twelsh@mitre.org',
)
