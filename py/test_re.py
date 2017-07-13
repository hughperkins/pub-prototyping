from __future__ import print_function
import re

target = 'foo bar -isysroot bar -I foobar -isysroot blah'

print(target)
print('exected: ', 'foo bar -I foobar')
print('actual:', re.sub(r' -isysroot [^ ]+', '', target))
