from __future__ import print_function

import os
import os.path
import sysconfig
import sys
import glob
import platform
from setuptools import setup
#from distutils.extension import Extension
from setuptools import Extension
from distutils.command.build_clib import build_clib
#import distutils.
from Cython.Build import cythonize

class mybuild(build_clib):
    def build_libraries(self,libraries):
        print('build_libraries')
        print(self.compiler.compiler_type)
        print(dir(self.compiler))
        #help(self.compiler.compile)
        #help(self.compiler.link_shared_lib)
        print('build_temp',self.build_temp)
        self.mkpath(self.build_temp)
        self.temp_src = self.build_temp + '/src'
        self.mkpath(self.temp_src)
        for libraryname, libdic in libraries:
            newsources = []
            print(libraryname)
            print(libdic['sources'])
            sources = libdic['sources']
            for source in sources:
                sourcefile = os.path.basename(source)
                print('source: ' + source )
#                help(self.compiler)
#                help(self.make_file)
#                self.make_file( source, sourcefile )
#                help(self.copy_file)
                self.copy_file( source, self.temp_src + '/' + sourcefile )
                newsources.append( self.temp_src + '/' + sourcefile )
#            libdic['sources'] = newsources
            objs = self.compiler.compile(newsources, '.' )
            print(objs)
            self.compiler.link_shared_lib(objs,libraryname,self.build_temp)
            libpath = self.compiler.library_filename(libraryname,'shared',0,self.build_temp )
            print('library path:',libpath )
            self.copy_file( libpath, '.' )
            print('outputdir',self.compiler.library_filename('blah'))
#        build_clib.build_libraries(self,libraries)
#    def get_source_files(self):
#        print('get_source_files')
    
#   def __init__(self,dist,foo):
#        super(mybuild,self).__init__(dist,foo)
#        print('mybuild')
#        print('dist',dist)

ext_modules = [
    Extension("pyfoo",
              sources=["pyfoo.pyx"], 
#                glob.glob('DeepCL/OpenCLHelper/*.h'),
              include_dirs = ['../src'],
              libraries= ['foo'],
#              extra_compile_args=compile_options,
#        define_macros = [('DeepCL_EXPORTS',1),('OpenCLHelper_EXPORTS',1)],
#              extra_objects=['cDeepCL.pxd'],
#              library_dirs = [lib_build_dir()],
              #runtime_library_dirs=['.'],
              language="c++"
    )
]

setup(
  name = 'test',
  libraries = [ ('foo', {'sources':['../src/foo.cpp']} )],
  ext_modules = cythonize(ext_modules),
  cmdclass = {'build_clib': mybuild }
)

