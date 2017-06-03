import tensorflow as tf
import ctypes

fact = ctypes.cdll.LoadLibrary('build/libfact.so')
fact.fact_init()
fact.init_cl_executor()
