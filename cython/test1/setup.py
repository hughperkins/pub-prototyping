import os
import sys
import sysconfig
from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize

# from http://stackoverflow.com/questions/14320220/testing-python-c-libraries-get-build-path
def distutils_dir_name(dname):
    """Returns the name of a distutils build directory"""
    f = "{dirname}.{platform}-{version[0]}.{version[1]}"
    return f.format(dirname=dname,
                    platform=sysconfig.get_platform(),
                    version=sys.version_info)
 
def lib_build_dir():
    return os.path.join('build', distutils_dir_name('lib'))

def get_so_suffix():
    if sysconfig.get_config_var('SOABI') != None:
        return "." + sysconfig.get_config_var('SOABI')
    return ""

def my_cythonize(extensions, **_ignore):
    newextensions = []
    for extension in extensions:
        print(extension.sources)
        should_cythonize = False
        sources = []
        for sfile in extension.sources:
            path, ext = os.path.splitext(sfile)
            if ext in ('.pyx', '.py'):
                should_cythonize = True
                if extension.language == 'c++':
                    ext = '.cpp'
                else:
                    ext = '.c'
                sfile = path + ext
            sources.append(sfile)
        print(should_cythonize)
        if should_cythonize:
            cythonize(extension)
        extension.sources[:] = sources    
        if should_cythonize:
            newextensions.append( extension )
        else:
            newextensions.append( extension )
    return newextensions

ext_modules = [
#    Extension("hello",
#              sources=["hello.pyx"],
#              libraries=["myc"]
#    ),
    Extension("libmycpp",
              sources=["mycpp.cpp"], 
              #library_dirs = [ lib_build_dir() ],
              #libraries=["myc"]
              language="c++"
    ),
#    Extension("pymyc",
#              sources=["pymyc.pyx"], 
#              library_dirs = [ lib_build_dir() ],
#              libraries=["myc"]
#    ),
    Extension("pymycpp",
              sources=["pymycpp.pyx"],
              library_dirs = [ lib_build_dir() ],
              libraries=["mycpp" + get_so_suffix()],
              runtime_library_dirs=["."],
              language="c++"
    )
]

setup(
  name = 'Hello world app',
  ext_modules = my_cythonize(ext_modules),
  scripts = ['run_hello.py']
)

