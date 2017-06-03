import time
import os
import curses

global win

def print(*args, **kwargs):
  mystr = ''
  for i, arg in enumerate(args):
    if i > 0:
      mystr += '\t'
    mystr += str(arg)
  win.addstr(0, 0, mystr)

# http://stackoverflow.com/questions/24072790/detect-key-press-in-python/32386410#32386410
def main(_win):
  global win
  win = _win
  win.nodelay(True)
  i = 0
  while True:
    key = None
    gotkey = True
    while gotkey:
      gotkey = False
      try:
        key = win.getkey()
        gotkey = True
        print('key', key)
      except:
        pass
      if gotkey:
        win.addstr(1, 0, key)
    print('dot!\n')
#    win.addstr(0, 0, 'dot %s!' % i)
    i += 1
    time.sleep(0.2)

curses.wrapper(main)

