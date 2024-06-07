from setuptools import setup, find_packages
from codecs import open
from os import path

__version__ = '3.4'

here = path.abspath(path.dirname(__file__))

# get the dependencies and installs
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    all_reqs = f.read().split('\n')

install_requires = [x.strip() for x in all_reqs if 'git+' not in x]
dependency_links = [x.strip().replace('git+', '') for x in all_reqs if x.startswith('git+')]

setup(
    name='mempred',
    version=__version__,
    description='Python module for performing time-series prediction using the generalized Langevin equation',
    #url='',
    #download_url='',
    #license='MIT',
    packages=find_packages(),
    include_package_data=True,
    authors='Henrik Kiefer, Niklas Kiefer',
    install_requires=install_requires,
    setup_requires=['yfinance','numpy', 'pandas', 'scipy', 'matplotlib', 'numba', 'wwo_hist','siml','sympy','tidynamics'],
    dependency_links=dependency_links,
    author_email='henrik.kiefer@fu-berlin.de'
)