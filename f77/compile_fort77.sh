#!/bin/bash

# make fortran library:
fort77 -c hpfn2.f && ar qc libhp.a hpfn2.o

# call fortran library from fortran:
fort77 -o hphello1 hphello1.f -L. -lhp && ./hphello1

# call fortran from c:
gcc -o callffromc callffromc.c -L. -lhp -lf2c && ./callffromc

# call fortran from cpp:
g++ -o callffromc callffromcpp.cpp -L. -lhp -lf2c && ./callffromc

