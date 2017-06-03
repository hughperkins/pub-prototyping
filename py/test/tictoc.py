
import time

start = 0

def tic():
    global start
    start = time.time()

def toc():
    global start
    newstart = time.time()
    print "Elapsed time: " + str((newstart - start) * 1000) + " ms"
    start = newstart


