from setuptools import find_packages, setup
from typing import List

HYPEN_DOT_E='-e .'
def get_requirements(file_path: str) -> list[str]:
    requirements=[]
    with open(file_path) as f:
        requirements=f.readlines()
        requirements=[req.replace("\n","") for req in requirements]
        
        if HYPEN_DOT_E in requirements:
            requirements.remove(HYPEN_DOT_E)
    return requirements

setup(
    name='my_package',
    version='0.1',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
    author='Ankit Gochhayat',
    author_email='ankit.gochhayat2004@gmail.com')