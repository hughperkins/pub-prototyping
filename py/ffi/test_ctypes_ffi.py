# from ctypes import cdll, c_int, c_float, c_char_p
import ctypes

lib = ctypes.cdll.LoadLibrary('build/libmylib.dylib')
lib.printName.restype = ctypes.c_char_p
print(lib.printName('hello', 123, ctypes.c_float(5.4)))
