# functors

From http://www.bogotobogo.com/cplusplus/functors.php

STL refines functor as follows:
- a generator is a functor that can be called with no argument
- a unary function is a functor that can be called with one argument
- a binary function is a functor that can be called with two arguments

Predicates are functions that return a boolean value, or something that can be impilcitly converted to `bool`.
- a unary function taht returns a bool value is a `predicate`
- a binary function taht returns a bool value is a `binary predicate`

Two types of `transform` algorithm:

transforming elements:
```
OutputIterator
transform(InputIterator source.begin(), InputIterator source.end(),
    OutputIterator destination.begin(),
    UnaryFunc op)
```

combining elements of two sequences
```
OutputIterator
transform(InputIterator1 source1.begin(), InputIterator1 source1.end(),
    InputIterator2 source2.begin(), InputIterator2 source2.end(),
    OutputIterator destination.begin(),
    BinaryFunc op)
```
