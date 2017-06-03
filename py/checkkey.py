import time
import sys
import fcntl
import termios
import tty
import os

# http://grokbase.com/t/python/python-list/0749p54xfs/check-for-keypress-on-linux-xterm
fd = sys.stdin.fileno()
oldterm = termios.tcgetattr(fd)
oldflags = fcntl.fcntl(fd, fcntl.F_GETFL)

tty.setcbreak(sys.stdin.fileno())
newattr = termios.tcgetattr(fd)
newattr[3] = newattr[3] & ~termios.ICANON & ~termios.ECHO

def newTerminalSettings():
  termios.tcsetattr(fd, termios.TCSANOW, newattr)
  fcntl.fcntl(fd, fcntl.F_SETFL, oldflags | os.O_NONBLOCK)

def oldTerminalSettings():
  termios.tcsetattr(fd, termios.TCSAFLUSH, oldterm)
  fcntl.fcntl(fd, fcntl.F_SETFL, oldflags)

newTerminalSettings()

def getlastchar():
  gotchar = True
  key = None
  while gotchar:
    gotchar = False
    try:
      newkey = sys.stdin.read(1)
      if newkey is not None:
        key = newkey
        gotchar = True
    except:
      pass
  return key

try:
  while True:
    c = getlastchar()
    if c is not None:
      print('c', c)
    print('dot!')
    time.sleep(0.2)
except:
  print('except')
  oldTerminalSettings()

