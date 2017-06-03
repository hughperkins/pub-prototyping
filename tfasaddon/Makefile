build/fact.o: fact.cpp
	g++ -std=c++11 -fPIC \
		-I$(HOME)/git/tensorflow-blas \
		-I$(HOME)/git/eigen \
		-I/norep/envs/env3-base-nocuda/lib/python3.5/site-packages/tensorflow/include \
		-DEIGEN_MPL2_ONLY -DEIGEN_AVOID_STL_ARRAY \
		-c $< -o $@

build/cl-%.o: cl/%.cc
	g++ -std=c++11 -fPIC \
	-I$(HOME)/git/tensorflow-blas \
	-I$(HOME)/git/eigen \
	-I$(HOME)/git/tensorflow-blas/third_party/cuda-on-cl/include/cocl \
	-I$(HOME)/git/tensorflow-blas/third_party/cuda-on-cl/src/EasyCL \
	-I/norep/envs/env3-base-nocuda/lib/python3.5/site-packages/tensorflow/include \
	-DEIGEN_MPL2_ONLY -DEIGEN_AVOID_STL_ARRAY \
	-c $< -o $@

build/common_runtime-gpu-%.o: common_runtime/gpu/%.cc
	g++ -std=c++11 -fPIC \
	-I$(HOME)/git/tensorflow-blas \
	-I$(HOME)/git/eigen \
	-I$(HOME)/git/tensorflow-blas/third_party/cuda-on-cl/include/cocl \
	-I$(HOME)/git/tensorflow-blas/third_party/cuda-on-cl/src/EasyCL \
	-I/norep/envs/env3-base-nocuda/lib/python3.5/site-packages/tensorflow/include \
	-DGOOGLE_CUDA \
	-DEIGEN_MPL2_ONLY -DEIGEN_AVOID_STL_ARRAY \
	-c $< -o $@

build/libfact.so:	build/fact.o build/cl-cl_blas.o build/cl-cl_gpu_executor.o build/cl-cl_platform.o \
		build/cl-cl_driver.o build/cl-cl_event.o build/cl-cl_stream.o build/cl-cl_platform_id.o \
		build/cl-cl_diagnostics.o build/cl-cl_blas.o build/cl-cl_activation.o
	g++ -o $@ -shared $^ \
		-Wl,-rpath=$(HOME)/git/tensorflow-blas/third_party/cuda-on-cl/build \
		-Wl,-rpath=$(HOME)/git/tensorflow-blas/third_party/cuda-on-cl/build/clblast \
		-L$(HOME)/git/tensorflow-blas/third_party/cuda-on-cl/build -lcocl -leasycl -lclew

#		build/common_runtime-gpu-gpu_device_factory.o
