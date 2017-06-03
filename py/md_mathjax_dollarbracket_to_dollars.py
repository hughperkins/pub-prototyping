infile = 'ibp_2.tex'
outfile = 'ibp_3.tex'

with open(infile, 'r') as f:
    contents = f.read()

pos = contents.find('\(')
while pos >= 0:
    print('pos', pos)
    print(contents[pos:pos + 2])
    contents = contents[:pos] + '$' + contents[pos + 2:]
    pos = contents.find('\(', pos)
    print('pos', pos)

pos = contents.find('\)')
while pos >= 0:
    print('pos', pos)
    print(contents[pos:pos + 2])
    contents = contents[:pos] + '$' + contents[pos + 2:]
    pos = contents.find('\)', pos)
    print('pos', pos)

with open(outfile, 'w') as f:
    f.write(contents)
