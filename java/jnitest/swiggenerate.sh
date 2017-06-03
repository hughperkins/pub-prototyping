#!/bin/bash


#mkdir -p swig/root/javalanguage/jni/swig
#swig -v -outdir root/javalanguage/jni/swig -java -package root.javalanguage.jni.swig -module JnitestC jnitest_c.h
mkdir -p swig/root
swig -v -outdir swig/root -java -package root jnitest_c.i



