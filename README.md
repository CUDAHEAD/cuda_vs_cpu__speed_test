# cuda_vs_cpu__speed_test

This simple code show how to get an acceleration of about x400 on AWS g2.2xlarge machine.
The idea is to run some simple code both on CPU and GPU. And messure the how fast can GPU get.

Prerequisite: Nvidia GPU, GPU drivers, GPU complier (nvcc)

To compile type: 
$ nvcc gpu_add_simple.cu

To run 
$ nvprof ./a.out 5120000 1024
