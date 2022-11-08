#include "virtual_memory.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

// __device__ void user_program(VirtualMemory *vm, uchar *input, uchar *results,int input_size, int thread_id) {
// 	printf("testing program begin\n");

// 	for(int i = 0; i < input_size;i++)
// 		vm_write(vm, i, input[i]);
	
// 	printf("part1 done\n");

// 	// for(int i = 0; i<)

//   	for (int i = input_size - 1; i >= input_size - 32769; i--)
//     	int value = vm_read(vm, i);
// 	// vm_read(vm, 0);
//   	vm_snapshot(vm, results, 0, input_size);
// 	// printf("testing program end\n");
// }

// __device__ void user_program(VirtualMemory *vm, uchar *input, uchar *results,int input_size, int thread_id) {
// 	// write the data.bin to the VM starting from address 32*1024
// 	for (int i = 0; i < input_size; i++)
// 	  	vm_write(vm, 32*1024+i, input[i]);
	
// 	// write (32KB-32B) data  to the VM starting from 0
// 	for (int i = 0; i < 32*1023; i++)
// 	  	vm_write(vm, i, input[i+32*1024]);
// 	// // readout VM[32K, 160K] and output to snapshot.bin, which should be the same with data.bin
// 	vm_snapshot(vm, results, 32*1024, input_size);
// }

// expected page fault num: 9215 =  4096 + 1023 + 4096

__device__ void user_program(VirtualMemory *vm, uchar *input, uchar *results, int input_size, int thread_id){
	printf("vm->storage_cnt = %d\n", vm->storage_cnt);

	printf("thread_id = %d\n",  thread_id);
	// input_size = 1024*32+32;
	for(int i = 0; i < input_size/4; i++){

		int addr = i + thread_id*(input_size/4);
		// printf("thread_id = %d addr = %d\n",thread_id, addr);
		if(thread_id == 0)
			vm_write(vm, addr, input[addr], thread_id);
		__syncthreads();
		if(thread_id == 1)
			vm_write(vm, addr, input[addr], thread_id);
		__syncthreads();
		if(thread_id == 2)
			vm_write(vm, addr, input[addr], thread_id);
		__syncthreads();
		if(thread_id == 3)
			vm_write(vm, addr, input[addr], thread_id);
		__syncthreads();
	}

	// for(int i = 0; i < input_size/4 ; i ++){

	// 	int addr = i + thread_id*(input_size/4);
	// 	if(thread_id == 0)
	// 		vm_read(vm, addr, thread_id);
	// 	__syncthreads();
	// 	if(thread_id == 1)
	// 		vm_read(vm, addr, thread_id);
	// 	__syncthreads();
	// 	if(thread_id == 2)
	// 		vm_read(vm, addr, thread_id);
	// 	__syncthreads();
	// 	if(thread_id == 3)
	// 		vm_read(vm, addr, thread_id);
	// 	__syncthreads();
	// }

	vm_snapshot(vm, results, 0, input_size, thread_id);
	// printf("page fault = %d\n", *(vm->pagefault_num_ptr));
}