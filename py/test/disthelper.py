#!/bin/python

import random
from scipy import array

def dirrnd(K, alpha):
    gammasamples = array([ random.gammavariate(alpha,1) for i in range(K) ])
    return gammasamples / sum(gammasamples)

if __name__ == '__main__':
    random.gammavariate(1,1)
    sample = dirrnd(3, 0.1)
    print sample
    print sum(sample)



