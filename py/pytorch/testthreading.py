import threading
import time
import PyTorch
import PyTorchHelpers
import numpy as np


#class MyClass(threading.Thread):
#  def __init__(self, name):
#    threading.Thread.__init__(self)
#    self.name = name
#    print('__init__', self.name)
#    time.sleep(1)
##    self.loop()

#  def run(self):
#   for i in range(30):
#     print(self.name, i)
#     time.sleep(1)

#pig = MyClass('pig')
#sheep = MyClass('sheep')
#dog = MyClass('dog')

#objs = []
#objs.append(pig)
#objs.append(sheep)
#objs.append(dog)

#for obj in objs:
#  obj.start()

#for obj in objs:
#  obj.join()

def dostuff(name):
  MyLuaClass = PyTorchHelpers.load_lua_class('testthreading.lua', 'MyLuaClass')
  print(name, 'dostuff start')
  obj = MyLuaClass(name)
  print('calling run', name)
  obj.run()
  print(name, 'dostuff done')

t = []
t.append(threading.Thread(target=dostuff, args=('pig',)))
t.append(threading.Thread(target=dostuff, args=('dog',)))
#t.append(threading.Thread(target=dostuff, args=('sheep',)))

for obj in t:
  obj.daemon = True
  obj.start()

#for obj in t:
#  obj.join()

time.sleep(30)

