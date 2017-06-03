--require 'torch'

--local TestStack = torch.class('TestStack')

--function TestStack:__init()
--  print('TestStack:__init()')
--end

STP = require 'StackTracePlus'

require 'debug'

--for k,v in pairs(debug) do
--  print('k', k)
--end

--debug.traceback = function(...)
--  print('debug.traceback()')
--end

function octopus(errmsg)
  print('errmsg', errmsg)
--  return debug.traceback()
--  local stackstr = debug.traceback()
--  print('traceback (in lua) ', stackstr)
----  return 'foo from octopus'
  local stackstr = STP.stacktrace()
  return stackstr
end

function foo(name)
  print('foo', name)
  error('bar')
  return "myresult"
end

function callfoo(name)
  local res = foo(name)
  return res
end

function callsub_anteater(name)
  local ok, res = xpcall(function()
    return callfoo(name)
  end, debug.traceback)
  local statusint = 0
  if not ok then
     statusint = -1
  end
  return statusint, res
end

