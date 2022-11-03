#include "virtual_memory.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

__device__ void init_invert_page_table(VirtualMemory *vm) {
	
  	for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
  	  	vm->invert_page_table[i] = 0x80000000; // invalid := MSB is 1
  	  	vm->invert_page_table[i + vm->PAGE_ENTRIES] = 0;
  	}
}

__device__ void vm_init(VirtualMemory *vm, uchar *buffer, uchar *storage,
                        u32 *invert_page_table, int *pagefault_num_ptr,
                        int PAGESIZE, int INVERT_PAGE_TABLE_SIZE,
                        int PHYSICAL_MEM_SIZE, int STORAGE_SIZE,
                        int PAGE_ENTRIES) {
  	// init variables
  	vm->buffer = buffer;
  	vm->storage = storage;
  	vm->invert_page_table = invert_page_table;
  	vm->pagefault_num_ptr = pagefault_num_ptr;
	
  	// init constants
  	vm->PAGESIZE = PAGESIZE;
  	vm->INVERT_PAGE_TABLE_SIZE = INVERT_PAGE_TABLE_SIZE;
  	vm->PHYSICAL_MEM_SIZE = PHYSICAL_MEM_SIZE;
  	vm->STORAGE_SIZE = STORAGE_SIZE;
  	vm->PAGE_ENTRIES = PAGE_ENTRIES;
	printf("vm->PAGESIZE = %d\n", PAGESIZE);

  	// before first vm_write or vm_read
  	init_invert_page_table(vm);
	
	// init LRU queue
	struct memory_item *tail;
	tail = (struct memory_item*)malloc(sizeof(memory_item));
	tail->up = NULL;
	tail->page_number = -1;			// 0 means no page 
	vm->LRU_bottom = tail;
	vm->LRU_top = tail;

	// initialize the physical memory count as 0
	vm->phyMem_cnt = 0;
}

__device__ void update_LRU(VirtualMemory *vm, int page_num){
	// update the stack 
	struct memory_item *temp = vm->LRU_bottom;
	struct memory_item *target = (struct memory_item*)malloc(sizeof(memory_item));
	// initialize the LRU_stack
	if(temp->page_number == -1){		// -1 means no page 
		// printf("LRU: first hit\n");
		vm->LRU_bottom->page_number = page_num;
		vm->LRU_top->page_number = page_num;
		return;
	}

	// if it is already the latest
	if(vm->LRU_top->page_number == page_num){
		// printf("LRU: no change\n");
		return;
	}

	// identify the memory_item and put it to the top of the stack
	while(temp->up != NULL){
		// identify the next item is the page
		if(temp->up->page_number == page_num){

			temp->up = temp->up->up;
			// step to the last item
			while(temp->up != NULL){
				temp = temp->up;
			}
			target->page_number = page_num;
			temp->up = target;
			vm->LRU_top = target;
			// printf("LRU: update\n");
			return;
		}

		// not reach the target yet...
		temp = temp->up;
	}
	// cannot find the item in the stack
	target->page_number = page_num;
	temp->up = target;
	vm->LRU_top = target;
	// printf("LRU: add new\n");
	return;
}

__device__ void delete_LRU(VirtualMemory * vm, int page_num){
	if(vm->LRU_bottom == NULL){
		// printf("LRU: empty\n");
		return;
	}
	struct memory_item *temp = vm->LRU_bottom;
	while(temp->up != NULL){
		if(temp->up->page_number == page_num){
			if(temp->up->up != NULL){
				temp->up = temp->up->up;
				// printf("LRU: delete 1\n");
				return;
			}
			else{
				temp->up = NULL;
				// printf("LRU: delete 2\n");
				return;
			}
		}
		temp = temp->up;
	}
	// printf("LRU: no delete\n");
	return;
}


// print the current sequence of LRU stack
__device__ void printLRUStack(VirtualMemory *vm){
	struct memory_item *temp = vm->LRU_bottom;
	while(temp!= NULL){
		if(temp->up == NULL)
			printf("%d \n ", temp->page_number);
		else
			printf("%d -> ", temp->page_number);
		temp = temp->up;
	}
	return;
}

__device__ uchar vm_read(VirtualMemory *vm, u32 addr) {
  	/* Complate vm_read function to read single element from data buffer */
	printf("read\n");

	int FRAME_SIZE = vm->PAGESIZE;
	int p = addr/FRAME_SIZE;
	int d = addr%FRAME_SIZE;
	int f = vm->invert_page_table[p];
	uchar out;
	update_LRU(vm, p);

	if(vm->invert_page_table[p+vm->PAGE_ENTRIES] == 0){
		vm->pagefault_num_ptr++;
		// perform page replacement
		printf(" perform page replacement\n");
		int victim_p = vm->LRU_bottom->page_number;
		int victim_f = vm->invert_page_table[victim_p];
		// step 1: swap out victim page
		for(int i = 0; i < FRAME_SIZE; i++){
			vm->storage[victim_f*FRAME_SIZE+i] = vm->buffer[victim_f*FRAME_SIZE+i];
		}
		// step 2: change to invalid
		vm->invert_page_table[victim_p] = 0;
		delete_LRU(vm, victim_p);
		// step 3: swap derised page in
		for (int i = 0; i < FRAME_SIZE; i++){
			vm->buffer[f*FRAME_SIZE+i] = vm->storage[f*FRAME_SIZE+i];
		}
		// read the value
		out = vm->buffer[f*FRAME_SIZE+d];
		// step 4: reset page table as valid for new page
		vm->invert_page_table[p+vm->PAGE_ENTRIES] = 1;
	}
	printf("read out = %d\n", out);
  	return out; 
}

__device__ void vm_write(VirtualMemory *vm, u32 addr, uchar value) {
  	/* Complete vm_write function to write value into data buffer */
	int FRAME_SIZE = vm->PAGESIZE;
	int p = addr/FRAME_SIZE;
	int d = addr%FRAME_SIZE;
	// periodic allocate the f to the p in page table
	int f = vm->PHYSICAL_MEM_SIZE%FRAME_SIZE;
	vm->invert_page_table[p] = f;
	// printf("p:%d, d:%d, f:%d\n", p, d, f);
	if(addr%1000 == 0)
		printf("addr = %d, vallue = %d\n", addr, value);

	
	// if the page entry is invalid
	update_LRU(vm, p);
	if(vm->invert_page_table[p+vm->PAGE_ENTRIES] == 0){
		vm->pagefault_num_ptr++;
		// perform page replacement
		int victim_p = vm->LRU_bottom->page_number;
		int victim_f = vm->invert_page_table[victim_p];
		// step 1: swap out victim page
		for(int i = 0; i < FRAME_SIZE; i++){
			vm->storage[victim_f*FRAME_SIZE+i] = vm->buffer[victim_f*FRAME_SIZE+i];
		}
		// step 2: change to invalid
		vm->invert_page_table[victim_p] = 0;
		// step 3: swap derised page in
		for (int i = 0; i < FRAME_SIZE; i++){
			vm->buffer[f*FRAME_SIZE+i] = value;
		}
		// write in new value
		vm->buffer[f+d] = value;
		// step 4: reset page table as valid for new page
		vm->invert_page_table[p+vm->PAGE_ENTRIES] = 1;
	}
}

__device__ void vm_snapshot(VirtualMemory *vm, uchar *results, int offset, int input_size) {
  	/* Complete snapshot function togther with vm_read to load elements from data
  	 * to result buffer */
}
