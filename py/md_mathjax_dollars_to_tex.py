infile = 'ibp.md'
outfile = 'ibp_2.md'

with open(infile, 'r') as f:
    contents = f.read()

i = 0

pos = contents.find('$$')
while pos >= 0:
    print('pos', pos)
    print(contents[pos:pos + 2])
    if i % 2 == 0:
        contents = contents[:pos] + '\\begin{equation}' + contents[pos + 2:]
    else:
        contents = contents[:pos] + '\\end{equation}' + contents[pos + 2:]
    i += 1
    pos = contents.find('$$', pos)
    print('pos', pos)
with open(outfile, 'w') as f:
    f.write(contents)
