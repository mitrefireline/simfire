import fnmatch
import os

import setuptools
from setuptools.command.build_py import build_py as build_py_orig

# Open the requirements to use as the package requirements
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# Get the correct COMMIT TAG to use for versioning
if os.environ.get('CI_COMMIT_TAG'):
    version = os.environ['CI_COMMIT_TAG']
else:
    version = '0.0.0'

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

# Package files to exclude (can use glob formatting)
excluded = ['**/game_rl.py', '**/harness.py']


# Define a new way of finding packages so that individual files can be excluded
class build_py(build_py_orig):
    def find_package_modules(self, package, package_dir):
        modules = super().find_package_modules(package, package_dir)
        return [
            (pkg, mod, file) for (pkg, mod, file) in modules
            if not any(fnmatch.fnmatchcase(file, pat=pattern) for pattern in excluded)
        ]


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
    cmdclass={'build_py': build_py},
    package_data={'assets': ['../assets/**/*']},
    author='The MITRE Corporation',
    author_email='twelsh@mitre.org',
)
