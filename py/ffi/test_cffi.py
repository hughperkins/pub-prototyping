from cffi import FFI

ffi = FFI()

mylib = ffi.dlopen('build/libmylib.dylib')
ffi.cdef("""
    const char * printName(const char * name, int count, float aFloat);
""")

print(ffi.string(mylib.printName('some name'.encode('utf-8'), 567, 7.89)).decode('utf-8'))

cocl = ffi.dlopen('/usr/local/lib/libcocl.so')

