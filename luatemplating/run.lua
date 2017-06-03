templater = require('templater')

local myenv = {}
myenv.color = 'red'
color = 'blue'
t = templater.compile_file('test1.cl', myenv)
print('t', t)

t = templater.compile_file('test1.cl', myenv)
print('t', t)

