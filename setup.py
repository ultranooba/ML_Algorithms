from setuptools import setup
import os
long_description = 'A simple module for using Machine Learning in your code.'
if os.path.exists('maclearn\README.md'):
    long_description = open('maclearn\README.md').read()

# This call to setup() does all the work
setup(
    name="MacLearn",
    version="1.1.1",
    description="A simple module for using Machine Learning in your code.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=['scikit-learn', 'pandas'],
    zip_safe = False,
    author="Sajedur Rahman Fiad",
    author_email="neural.gen.official@gmail.com",
    packages = ['maclearn']
)
