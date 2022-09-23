#!/bin/bash
clang++ -c main.cpp complex.cpp
nvcc -c matmul.cu cudaft.cu
clang++ -L/usr/local/cuda/lib64 main.o complex.o matmul.o cudaft.o -lcudart -lraylib -lGL -lm -lpthread -ldl -lrt -lX11
