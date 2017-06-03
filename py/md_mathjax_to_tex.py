import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--infile', type=str, required=True, help='something.md')
parser.add_argument('--outfile', type=str, required=True, help='something.md')
args = parser.parse_args()

infile = args.infile
outfile = args.outfile

with open(infile, 'r') as f:
    contents = f.read()

i = 0


def replace(contents, find, replace):
    pos = contents.find(find)
    while pos >= 0:
        print('pos', pos)
        print(contents[pos:pos + len(find)])
        contents = contents[:pos] + replace + contents[pos + len(find):]
        pos = contents.find(find, pos)
        print('pos', pos)
    return contents

r = {'\gt': '>'}
for before, after in r.items():
    contents = replace(contents, before, after)

"""
handle image references, since they get moved out of their inline position
image refs look like:

![png](output_42_0.png)

and we want to change to something like:

See Figure \ref{output_42_0}

![\label{output_42_0}](output_42_0.png)

"""

pos = contents.find('![png](')
while pos >= 0:
    prefix = contents[:pos]
    target_length = contents[pos:].find(')') + 1
    print('length', target_length)
    postfix = contents[pos + target_length:]
    print('target[:20]', contents[pos:pos + 40])
    print('postfix[:20]', postfix[:20])
    target = contents[pos:pos + target_length]
    print('target', target)
    ref_string = target.split('(')[1].split('.')[0]
    new_target = 'See Figure \\ref{%s}\n\n![\\label{%s}](%s.png)' % (
        ref_string, ref_string, ref_string)
    print('new_target [%s]' % new_target)
    contents = prefix + new_target + postfix
    pos = contents.find('![png](')

with open(outfile, 'w') as f:
    f.write(contents)
