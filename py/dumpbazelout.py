"""
takes a linker command from tf build, and lists just the names, without hte paths, and split onto newlines
"""
import argparse


def run(filepath):
    with open(filepath, 'r') as f:
        contents = f.read().replace('\n', '')
    libs = []
    for line in contents.split():
        line = '/'.join(line.split('/')[-2:])
        if line == '':
            continue
        if line.startswith('-Wl'):
            continue
        # if not line.startswith('lib'):
        #     continue
        # line = line[3:]
        libs.append(line)
    libs.sort()
    for lib in libs:
        print(lib)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', type=str, required=True)
    args = parser.parse_args()
    run(**args.__dict__)
