require 'torch'
local ffi = require('ffi')

ffi.cdef[[
void say_hello();
void say(char *name);
void analyse(THFloatTensor *tensor);
]]
local foo = ffi.load('foo')

foo.say_hello();
local name = 'bar'
foo.say(ffi.new('char[?]', name:len(), name));
a = torch.FloatTensor(3,2):uniform()
foo.analyse(a:cdata())
print('a', a)

