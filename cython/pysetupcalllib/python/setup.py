# Copyright Hugh Perkins 2015 hughperkins at gmail
#
# This Source Code Form is subject to the terms of the Mozilla Public License, 
# v. 2.0. If a copy of the MPL was not distributed with this file, You can 
# obtain one at http://mozilla.org/MPL/2.0/.

from __future__ import print_function
import os
import os.path
import sysconfig
import sys
import glob
import platform
from setuptools import setup
from setuptools import Extension

cython_available = False
try:
    from Cython.Build import cythonize
    cython_available = True
except:
    print('warning: cython not available')

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

compile_options = []
osfamily = platform.uname()[0]
if osfamily == 'Windows':
   compile_options.append('/EHsc')
elif osfamily == 'Linux':
   compile_options.append('-std=c++0x')
   compile_options.append('-g')
else:
   pass
   # put other options etc here if necessary

runtime_library_dirs = []
libraries = []
libraries.append('myclib')

if osfamily == 'Linux':
    runtime_library_dirs= ['.']

if osfamily == 'Windows':
    libraries.append('winmm')

sources = ['testpylib.pyx']
if not cython_available:
    source = ['testpylib.cpp']
ext_modules = [
    Extension("testpylib",
              sources=sources,
              include_dirs = ['.'],
              libraries= libraries,
              extra_compile_args=compile_options,
              runtime_library_dirs=runtime_library_dirs,
              language="c++"
    )
]

if cython_available:
    ext_modules = cythonize(ext_modules)

def read_if_exists(filename):
    filepath = os.path.join(os.path.dirname(__file__), filename)
    if os.path.isfile(filepath):
        return open(filepath).read()
    else:
        return ""

setup(
  name = 'testpylib',
  version = '0.0',
  author = "",
  author_email = "",
  description = '',
  license = '',
  url = '',
  long_description = '',
  classifiers = [
  ],
  install_requires = [],
  scripts = [],
  ext_modules = ext_modules,
)

