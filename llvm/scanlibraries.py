"""
Find which library has a symbol
"""
import argparse
import subprocess
import os
from os.path import join


def search_lib(clang_home, libname, symbol):
    fout = open('/tmp/scanout.txt', 'w+')
    libpath = join(clang_home, 'lib', libname)
    res = subprocess.run([
        # 'ar', '-t', libpath
        'nm', '-C', libpath
    ], stdout=fout, stderr=subprocess.STDOUT)
    # print(' '.join(res.args))
    # fout.close()
    fout.seek(0)
    output = fout.read()
    # print(output)
    assert res.returncode == 0
    if symbol in output:
        print(libpath)
        for line in output.split('\n'):
            if symbol in line and :
                print('    ', line)


def run(clang_home, symbol):
    for file in os.listdir(join(clang_home, 'lib')):
        if file.endswith('.a'):
            search_lib(clang_home, file, symbol)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clang_home', default='/usr/lib/llvm-3.8')
    parser.add_argument('--symbol', type=str, required=True)
    args = parser.parse_args()
    run(**args.__dict__)
