from setuptools import find_packages,setup

def get_requirements(file_path:str): 
    'this function will return the list of requirements'
    Hyphen_E='-e .'
    with open (file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace('\n','') for req in requirements]
        
    if Hyphen_E in requirements:
        requirements.remove(Hyphen_E)
    return requirements
setup(

name='mlproject',
version='0.0.1',
author='Faye Sharafkhani',
author_email='fa.sharafkhani243@gmail.com',
packages=find_packages(),
install_requires=get_requirements('requirements.txt')


)