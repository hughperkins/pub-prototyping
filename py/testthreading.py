import threading
import time

class MyClass(threading.Thread):
  def __init__(self, name):
    threading.Thread.__init__(self)
    self.name = name
    print('__init__', self.name)
    time.sleep(1)
#    self.loop()

  def run(self):
   for i in range(30):
     print(self.name, i)
     time.sleep(1)

pig = MyClass('pig')
sheep = MyClass('sheep')
dog = MyClass('dog')

objs = []
objs.append(pig)
objs.append(sheep)
objs.append(dog)

for obj in objs:
  obj.start()

for obj in objs:
  obj.join()

