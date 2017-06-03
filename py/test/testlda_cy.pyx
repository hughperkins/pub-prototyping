#!/bin/python

from __future__ import division
import random
import scipy
from numpy.random import dirichlet, multinomial, randint
from scipy import array, zeros, bincount, shape, logical_and, single, double
from tictoc import tic, toc
import numpy
cimport numpy
import cython
#from disthelper import dirrnd

numpy.import_array()

# let's first create some random topics
# then mix them to create samples

def discreternd(numpy.ndarray[numpy.double_t, ndim=1] probabilities not None):
    cdef float p = random.random()
    cdef float total = 0
    cdef int i = 0
    cdef int N = len(probabilities)
    for i in xrange(N):
        total += probabilities[i]
        if p <= total:
            return i
    return N - 1

def mysumd(numpy.ndarray[numpy.double_t,ndim=1] inarray not None):
   cdef float total
   cdef float value
   for value in inarray:
      total += value
   return value

def mydivided1( numpy.ndarray[numpy.double_t,ndim=1] one not None, double s ):
    cdef int N = len(one)
    cdef numpy.ndarray[numpy.double_t] result = zeros(N, double )
    cdef int i
    for i in xrange(N):
        result[i] = one[i] / s
    return result

@cython.boundscheck(False)
def go():
   #sampleAlpha = 0.3
   cdef float alpha = 0.1
   cdef float beta = 0.1

   cdef int N = 100 # num docs
   cdef int M = 30 # num words per doc
   cdef int J = 5 # vocab size

   cdef int burnin=30
   cdef int samples=5

   cdef int K = 3
   cdef numpy.ndarray[numpy.double_t, ndim=2] topics = array([
       [0.5, 0.5, 0, 0, 0],
       [0.5, 0, 0, 0, 0.5],
       [0, 0.333, 0.333, 0.333, 0 ]
   ])
   print topics

   cdef int it
   cdef int n
   cdef int m
   cdef int k

   iterations = burnin + samples

   cdef numpy.ndarray[numpy.int_t, ndim=2] X = zeros([N,M], int)
   for n in xrange(N):
       Theta = dirichlet( [alpha] * K )
       for m in xrange(M):
           thisz = discreternd(Theta)
           thisworddist = topics[thisz]
           thisword = discreternd(thisworddist )
           X[n,m] = thisword

   print X

   # now do the whole lda thing...
   # 'X' contains the list of words assigned to each doc

   # first arbitrarily assign a topic to each word
   cdef numpy.ndarray[numpy.int_t, ndim=2] Z = randint(0,K,[N,M])

   # now calculate sums
   # we need: c_kj c_kn c_k

   tic()
   print type(Z.flatten())
   print type(Z.flatten()[0])
   print type(bincount(Z.flatten()))
   print type(bincount(Z.flatten()).astype(double))
   cdef numpy.ndarray[numpy.int_t, ndim=1] c_k = bincount(Z.flatten())
   cdef numpy.ndarray[numpy.int_t, ndim=2] c_nk = zeros([N,K], int)
   cdef numpy.ndarray[numpy.int_t, ndim=2] c_jk = zeros([J,K], int)
   for n in xrange(N):
       c_nk[n] = bincount(Z[n], minlength=K)
   for j in xrange(J):
       for k in xrange(K):
           c_jk[j,k] = shape(logical_and(Z == k, X == j ).nonzero())[1]
   toc()
   print c_k
   print c_jk
   print c_nk

   # now iterate...

   avg_c_k = zeros(K)
   avg_c_jk = zeros([J,K])
   tic()
   cdef int oldk
   cdef int newk
   cdef int thisj
   cdef numpy.ndarray[numpy.double_t, ndim=1] prob_by_k
   prob_by_k = zeros(K)
   for it in xrange(iterations):
       print "it: " + str(it)
       for n in xrange(N):
           for m in xrange(M):
              oldk = Z[n,m]
              thisj = X[n,m]
              c_k[oldk] -= 1
              c_nk[n,oldk] -= 1
              c_jk[thisj,oldk] -= 1
              
              for k in xrange(K):
                 prob_by_k[k] = ( ( c_nk[n,k] + alpha ) * (c_jk[thisj,k] + beta ) 
                     / ( c_k[k] + J * beta ) )
              prob_by_k = mydivided1( prob_by_k, mysumd(prob_by_k) )
              newk = discreternd(prob_by_k)
              
              Z[n,m] = newk
              c_k[newk] += 1
              c_nk[n,newk] += 1
              c_jk[thisj,newk] += 1
       if it > burnin:
           avg_c_k += c_k / samples
           avg_c_jk += c_jk / samples
       toc()

   p_given_k_of_j = zeros([K,J])
   for k in xrange(K):
       for j in xrange(J):
           p_given_k_of_j[k,j] = ( avg_c_jk[j,k] + beta ) / ( avg_c_k[k] + J * beta )
   print p_given_k_of_j
   print p_given_k_of_j > 0.1

go()

