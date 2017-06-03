import sys
import threading
import time


data_ready = threading.Event()

# https://mail.python.org/pipermail/tutor/2001-November/009936.html
class KeyboardPoller( threading.Thread ) :
    def run( self ) :
        global key_pressed
        while True:
          print('waiting on ch...')
          ch = sys.stdin.read( 1 ) 
          print('got ch', ch)
          if ch == 'K' : # the key you are interested in
              key_pressed = 1
          else :
              key_pressed = 0
          data_ready.set()

poller = KeyboardPoller()
poller.daemon = True
poller.start()

while True:
  if data_ready.isSet():
     print('K pressed')
  print('dot!')
  time.sleep(0.2)


