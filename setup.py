from setuptools import setup, find_packages
#from pip.req import parse_requirements

#install_reqs = parse_requirements('requirements.txt', session='hack')
#reqs = [str(ir.req) for ir in install_reqs]
reqs=[]
setup(
    name='streamlit',
    description='Various NLP topics',
    author='Alex V.',
    version='0.1.0',
    packages= find_packages(),
    install_requires=reqs  
)