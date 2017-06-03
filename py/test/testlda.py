#!/bin/python

from __future__ import division
import random
import scipy
from numpy.random import dirichlet, multinomial, randint
from scipy import array, zeros, bincount, shape, logical_and
from tictoc import tic, toc
#from disthelper import dirrnd

# let's first create some random topics
# then mix them to create samples

def discreternd2(probabilities):
    return multinomial(1,probabilities).nonzero()[0][0]

def discreternd(probabilities):
    p = random.random()
    total = 0
    i = 0
    N = len(probabilities)
    for i in xrange(N):
        total += probabilities[i]
        if p <= total:
            return i
    return N - 1

#sampleAlpha = 0.3
alpha = 0.1
beta = 0.1

N = 100 # num docs
M = 30 # num words per doc
J = 5 # vocab size

burnin=30
samples=5

K = 3
topics = array([
    [0.5, 0.5, 0, 0, 0],
    [0.5, 0, 0, 0, 0.5],
    [0, 0.333, 0.333, 0.333, 0 ]
])
print topics

iterations = burnin + samples

X = zeros([N,M],'float')
for n in xrange(N):
    Theta = dirichlet( [alpha] * K )
    for m in xrange(M):
        thisz = discreternd(Theta)
        #print '1'
        #print topics
        #print thisz
        thisworddist = topics[thisz]
        #print thisworddist
        #print topics[thisz]
        thisword = discreternd(thisworddist )
        X[n,m] = int(thisword);

print X

# now do the whole lda thing...
# 'X' contains the list of words assigned to each doc

# first arbitrarily assign a topic to each word
Z = randint(0,K,[N,M])

# now calculate sums
# we need: c_kj c_kn c_k

tic()
c_k = bincount(Z.flatten())
c_nk = zeros([N,K], 'float')
c_jk = zeros([J,K], 'float')
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

avg_c_k = zeros(K,'float')
avg_c_jk = zeros([J,K],'float')
tic()
for it in xrange(iterations):
    print "it: " + str(it)
    for n in xrange(N):
        for m in xrange(M):
           oldk = Z[n,m]
           thisj = X[n,m]
           c_k[oldk] -= 1
           c_nk[n,oldk] -= 1
           c_jk[thisj,oldk] -= 1
           
           prob_by_k = zeros(K,'float')
           for k in xrange(K):
              prob_by_k[k] = ( ( c_nk[n,k] + alpha ) * (c_jk[thisj,k] + beta ) 
                  / ( c_k[k] + J * beta ) )
           prob_by_k = prob_by_k / sum(prob_by_k)
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


