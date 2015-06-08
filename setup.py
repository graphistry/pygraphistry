"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

from setuptools import setup, find_packages
#from codecs import open
#from os import path

#here = path.abspath(path.dirname(__file__))

#with open(path.join(here, 'DESCRIPTION.rst'), encoding='utf-8') as f:
 #   long_description = f.read()

setup(
    name='Graphistry',

    version='1.1.0.dev1',
    py_modules = ['graphistry'],
    description = 'This is established as a Data Loader for Graphistry',
    #long_description=long_description,

    url='https://github.com/graphistry/pygraphistry',

    author='Graphistry',
    author_email='xin@graphistry.com',

    license='MIT',

    classifiers=[
        'Development Status :: 3 - Alpha',

        'Intended Audience :: Graphistry User',
        'Topic :: Data Visualization Development :: Load Tools',

        'License :: OSI Approved :: MIT License',

        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
    ],

    keywords='Python Data Loader',

    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),


)
