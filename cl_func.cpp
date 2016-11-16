#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <cassert>

#include "util_cl.h"

using namespace std;

//#define CPU 1
#define GPU 1
#define THREAD_NUM 100000

namespace OpenCL {

	unsigned test(const unsigned simulate_total)
	{
		cl_int err;
		cl_event event;

		cl_platform_id platform;
		err = clGetPlatformIDs(1, &platform, NULL);
		cout << "The 1st Platform: " << GetPlatformName(platform) << endl;

		cl_uint cpuDeviceCount;
		err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 0, NULL, &cpuDeviceCount);

		//get all CPU devices
		cl_device_id* cpuDevices;
		cpuDevices = new cl_device_id[cpuDeviceCount];
		err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, cpuDeviceCount, cpuDevices, NULL);

		for (cl_uint i = 0; i < cpuDeviceCount; ++i) {
			cout << "\t CPU (" << (i + 1) << ") : " << GetDeviceName (cpuDevices[i]) << endl;
		}

		// for each cpu device create a separate context AND queue
		cl_context* cpuContexts = new cl_context[cpuDeviceCount];
		cl_command_queue* cpuQueues = new cl_command_queue[cpuDeviceCount];
		for (int i = 0; i < cpuDeviceCount; i++) {
			cpuContexts[i] = clCreateContext(NULL, cpuDeviceCount, cpuDevices, NULL, NULL, &err);
			cpuQueues[i] = clCreateCommandQueue(cpuContexts[i], cpuDevices[i], CL_QUEUE_PROFILING_ENABLE, &err);
		}

		//get GPU device count
		cl_uint gpuDeviceCount;
		err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &gpuDeviceCount);

		//get all GPU devices
		cl_device_id* gpuDevices;
		gpuDevices = new cl_device_id[gpuDeviceCount];
		err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, gpuDeviceCount, gpuDevices, NULL);

		for (cl_uint i = 0; i < gpuDeviceCount; ++i) {
			cout << "\t GPU (" << (i + 1) << ") : " << GetDeviceName (gpuDevices[i]) << endl;
		}

		// for each Gpu device create a separate context AND queue
		cl_context* gpuContexts = new cl_context[gpuDeviceCount];
		cl_command_queue* gpuQueues = new cl_command_queue[gpuDeviceCount];
		for (int i = 0; i < gpuDeviceCount; i++) {
			gpuContexts[i] = clCreateContext(NULL, gpuDeviceCount, gpuDevices, NULL, NULL, &err);
			gpuQueues[i] = clCreateCommandQueue(gpuContexts[i], gpuDevices[i], CL_QUEUE_PROFILING_ENABLE, &err);
		}

#ifdef CPU
	cout << "OpenCL CPU: " << endl;
	cl_program cpuProgram = CreateProgram(cpuContexts[0], cpuDevices[0], "kernel.cl");
	cl_kernel cpuKernel = clCreateKernel(cpuProgram, "compute" ,&err);
	err_check(err, "cpuKernel");	

	// each thread has one local counter.
	vector<unsigned int> h_count(THREAD_NUM, 0);
	cout << "Thread num: " << THREAD_NUM << endl;

	unsigned int simulate_local = simulate_total / THREAD_NUM;
	assert((simulate_total%THREAD_NUM == 0) && "cannot divide perfectly");
	
	cl_mem d_count = clCreateBuffer(cpuContexts[0], CL_MEM_COPY_HOST_PTR,
			sizeof(unsigned int) * h_count.size(), &h_count[0], &err);
	err_check(err, "clCreateBuffer d_A");

	// set kernel arguments
	err = clSetKernelArg(cpuKernel, 0, sizeof(cl_mem), &d_count);
	err |= clSetKernelArg(cpuKernel, 1, sizeof(unsigned int), &simulate_local);
	//err |= clSetKernelArg(cpuKernel, 2, sizeof(float), &c);
	err_check(err, "set Kernel Arg");

	clFinish(cpuQueues[0]);
	
	// Launch kernel
	size_t globalWorkSize[] = {THREAD_NUM};
	err = clEnqueueNDRangeKernel(cpuQueues[0], cpuKernel, 1, NULL, globalWorkSize, NULL,
			0, NULL, &event);
	err_check(err, "clEnqueueNDRangeKernel");

	clWaitForEvents(1, &event);
	
	// copy back data from device
	err = clEnqueueReadBuffer(cpuQueues[0], d_count, CL_TRUE, 0, sizeof(unsigned int) * h_count.size(), &h_count[0],
			0, NULL, NULL);
	err_check(err, "copy back ");

	// count reduction.
	unsigned int total_count = 0;
	for(int i = 0; i < THREAD_NUM; ++i)
		total_count += h_count[i];
#endif

#ifdef GPU
	cout << "OpenCL GPU: " << endl;
	cl_program gpuProgram = CreateProgram(gpuContexts[0], gpuDevices[0], "kernel.cl");
	cl_kernel gpuKernel = clCreateKernel(gpuProgram, "compute" ,&err);
	err_check(err, "gpuKernel");	

	// each thread has one local counter.
	vector<unsigned int> h_count(THREAD_NUM, 0);
	cout << "Thread num: " << THREAD_NUM << endl;

	unsigned int simulate_local = simulate_total / THREAD_NUM;
	assert((simulate_total%THREAD_NUM == 0) && "cannot divide perfectly");
	
	cl_mem d_count = clCreateBuffer(gpuContexts[0], CL_MEM_COPY_HOST_PTR,
			sizeof(unsigned int) * h_count.size(), &h_count[0], &err);
	err_check(err, "clCreateBuffer d_A");

	// set kernel arguments
	err = clSetKernelArg(gpuKernel, 0, sizeof(cl_mem), &d_count);
	err |= clSetKernelArg(gpuKernel, 1, sizeof(unsigned int), &simulate_local);
	//err |= clSetKernelArg(gpuKernel, 2, sizeof(float), &c);
	err_check(err, "set Kernel Arg");

	clFinish(gpuQueues[0]);
	
	// Launch kernel
	size_t globalWorkSize[] = {THREAD_NUM};
	err = clEnqueueNDRangeKernel(gpuQueues[0], gpuKernel, 1, NULL, globalWorkSize, NULL,
			0, NULL, &event);
	err_check(err, "clEnqueueNDRangeKernel");

	clWaitForEvents(1, &event);
	
	// copy back data from device
	err = clEnqueueReadBuffer(gpuQueues[0], d_count, CL_TRUE, 0, sizeof(unsigned int) * h_count.size(), &h_count[0],
			0, NULL, NULL);
	err_check(err, "copy back ");

	// count reduction.
	unsigned int total_count = 0;
	for(int i = 0; i < THREAD_NUM; ++i)
		total_count += h_count[i];
#endif

	// get the profiling data
	cl_ulong timeStart, timeEnd;
	double totalTime;
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(timeStart), &timeStart, NULL);
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(timeEnd), &timeEnd, NULL);
	totalTime = timeEnd - timeStart;
	cout << "\nKernel Execution time in milliseconds = " << totalTime/1000000.0 << " ms\n" << endl;

	// cleanup CPU
	for(int i = 0; i < cpuDeviceCount; i++) {
#ifdef __APPLE__
		clReleaseDevice(cpuDevices[i]);
#endif
		clReleaseContext(cpuContexts[i]);
		clReleaseCommandQueue(cpuQueues[i]);
	}

	delete[] cpuDevices;
	delete[] cpuContexts;
	delete[] cpuQueues;

	// cleanup GPU
	for(int i = 0; i < gpuDeviceCount; i++) {
#ifdef __APPLE__
		clReleaseDevice(gpuDevices[i]);
#endif
		clReleaseContext(gpuContexts[i]);
		clReleaseCommandQueue(gpuQueues[i]);
	}

	delete[] gpuDevices;
	delete[] gpuContexts;
	delete[] gpuQueues;

	return total_count;
	}
}
