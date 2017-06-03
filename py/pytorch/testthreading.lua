require 'sys'
require 'os'
require 'torch'

local MyLuaClass = torch.class('MyLuaClass')

function MyLuaClass.__init(self, name)
  self.name = name
  print(self.name, '__init')
end

function MyLuaClass.run(self)
  print(self.name, 'start')
  sys.execute('sleep 3')
  print(self.name, 'finish')
end

