import numpy as np
import pickle
import platform
import os
from os.path import join
#import scipy
#from scipy import ndimage
from scipy import misc
pyversion = int(platform.python_version_tuple()[0])
if pyversion == 2:
  import cPickle
else:
  import pickle

data_dir = '/home/ubuntu/git/pytorch-residual-networks/cifar-10-batches-py'

def loadPickle(path):
  with open(path, 'rb') as f:
    if pyversion == 2:
      return cPickle.load(f)
    else:
      return {k.decode('utf-8'): v for k,v in pickle.load(f, encoding='bytes').items()}

inputPlanes = 3
inputWidth = 32
inputHeight = 32

d = loadPickle(join(data_dir, 'test_batch'))
NTest = d['data'].shape[0]
testData = np.zeros((NTest, inputPlanes, inputWidth, inputHeight), np.float32)
testLabels = np.zeros(NTest, np.uint8)
data = d['data'].reshape(NTest, inputPlanes, inputWidth, inputHeight)
testData[:] = data
testLabels[:] = d['labels']

i = 4
print('testData[i].shape', testData[i].shape)
print('testData[i].transpose(1,2,0).shape', testData[i].transpose(1,2,0).shape)
misc.imsave('/tmp/test.png', testData[i].transpose(1,2,0))

sample = testData[i]
sample = np.fliplr(sample.transpose(1,2,0)).transpose(2,0,1)
misc.imsave('/tmp/test2.png', sample.transpose(1,2,0))

sample2 = np.zeros((inputPlanes, inputWidth, inputHeight), np.float32)
range1 = 2:20
sample2[:,range1,2:20] = sample[:,2:20,2:20]
misc.imsave('/tmp/test3.png', sample2.transpose(1,2,0))

