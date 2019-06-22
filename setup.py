import os 
from setuptools import setup, find_packages


#Use README file for long description of the project
##     RootDir
###    - README.md
###    - setup.py

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

def parse_requiremetns(fname):
    with open(fname) as f:
        required = f.read().splitlines()
    print(required)
    return required
    
setup(
    name = "NavigateAgent",
    author = "Akhil Singh Rana",
    author_email = "er.akhil.singh.rana@gmail.com",
    description = ("This is a project from Udacity nanodegree program"
                    "Advanced  Reinforcement Learning"),
    long_description = read("README.md"),
    install_requires = parse_requiremetns("requirements.txt"),    
    packages=find_packages(),

)