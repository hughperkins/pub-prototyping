from __future__ import print_function, division


class Foo(object):
    def __truediv__(self, second):
        print('__truediv__')

    def __itruediv__(self, second):
        print('__itruediv__')

    def __floordiv__(self, second):
        print('__floordiv__')

    def __floordiv__(self, second):
        print('__floordiv__')

foo = Foo()
foo / 3
foo /= 3
# foo // 3
# foo //= 3
