#include "virtual_memory.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

__device__ void init_invert_page_table(VirtualMemory *vm) {
	
  	for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
		// the frame index is i
  	  	vm->invert_page_table[i] = -1;		
  	}
}

__device__ void init_swap_table(VirtualMemory *vm){

	for (int i = 0; i < vm->STORAGE_SIZE+vm->PHYSICAL_MEM_SIZE; i++){
		vm->swap_table[i] = -1;
	}
}

__device__ void vm_init(VirtualMemory *vm, uchar *buffer, uchar *storage,
                        int *invert_page_table, int *pagefault_num_ptr,
						int *swap_table,
                        int PAGESIZE, int INVERT_PAGE_TABLE_SIZE,
                        int PHYSICAL_MEM_SIZE, int STORAGE_SIZE,
                        int PAGE_ENTRIES) {
  	// init variables
  	vm->buffer = buffer;
  	vm->storage = storage;
  	vm->invert_page_table = invert_page_table;
  	vm->pagefault_num_ptr = pagefault_num_ptr;
	vm->swap_table = swap_table;
	
  	// init constants
  	vm->PAGESIZE = PAGESIZE;
  	vm->INVERT_PAGE_TABLE_SIZE = INVERT_PAGE_TABLE_SIZE;
  	vm->PHYSICAL_MEM_SIZE = PHYSICAL_MEM_SIZE;
  	vm->STORAGE_SIZE = STORAGE_SIZE;
  	vm->PAGE_ENTRIES = PAGE_ENTRIES;
	printf("vm->PAGESIZE = %d\n", PAGESIZE);

  	// before first vm_write or vm_read
  	init_invert_page_table(vm);

	init_swap_table(vm);

	
	// init LRU queue
	struct memory_item *tail;
	tail = (struct memory_item*)malloc(sizeof(memory_item));
	tail->up = NULL;
	tail->page_number = -1;			// 0 means no page 
	vm->LRU_bottom = tail;
	vm->LRU_top = tail;

	// initialize the swap table allocate storage count as 0
	vm->storage_cnt = 0;
}

__device__ void showLRU(VirtualMemory *vm){
	struct memory_item *temp = vm->LRU_bottom;
	while(temp->up != NULL){
		printf("%d ->", temp->page_number);
		temp = temp->up;
	}
	printf("%d\n", temp->page_number);
}

__device__ void update_LRU(VirtualMemory *vm, int page_num){		// put the page_num to the top of the stack

	struct memory_item *temp = vm->LRU_bottom;
	struct memory_item *target = (struct memory_item*)malloc(sizeof(memory_item));

	if(vm->LRU_bottom->page_number == -1){							// initialize the LRU_bottom
		vm->LRU_bottom->page_number = page_num;
		vm->LRU_top->page_number = page_num;
	}
	else if(vm->LRU_top->page_number == page_num){					// already top
		return;	
	}
	else if(vm->LRU_bottom->page_number == page_num){				// remove bottom to top
		vm->LRU_bottom = vm->LRU_bottom->up;
		vm->LRU_top = temp;
		while(temp->up != NULL){
			temp = temp->up;
		}
		target->page_number = page_num;
		temp->up = target;
		// showLRU(vm);
	}
	else{															// find the target and remove to top
		while(temp->up != NULL){
			if(temp->up->page_number == page_num){
				target = temp->up;
				temp->up = target->up;
				temp = temp->up;
				target->up = NULL;
				while(temp->up !=NULL){
					temp = temp->up;
				}
				temp->up = target;
				vm->LRU_top = target;
				// showLRU(vm);
				return;
			}
			temp = temp->up;
		} 
		target->page_number = page_num;
		target->up = NULL;
		temp->up = target;
		vm->LRU_top = target;
		// showLRU(vm);
	}
}

__device__ int get_frameIdx(VirtualMemory *vm, int page_num){
	// find frame index of the given page number in the page table
	for(int i = 0; i < vm->PAGE_ENTRIES; i++){
		if(vm->invert_page_table[i] == page_num){
			return i;
		}
		else if(vm->invert_page_table[i] == -1){		// the page entry is not used
			(*(vm->pagefault_num_ptr))++;
			return i;
		}
	}
	return -1;		// -1 if the page number isn't exist in the page table
}

__device__ uchar vm_read(VirtualMemory *vm, u32 addr) {
  	/* Complate vm_read function to read single element from data buffer */
	uchar out;	
	int p = addr/vm->PAGESIZE;
	int d = addr%vm->PAGESIZE;
	int f = get_frameIdx(vm, p);

	if(f == -1){
		(*(vm->pagefault_num_ptr))++;
		int victim_p = vm->LRU_bottom->page_number;
		vm->LRU_bottom = vm->LRU_bottom->up;											// pop out
		int victim_f = get_frameIdx(vm, victim_p);
		// printf("victim_f = %d\n", victim_f);
		int s_in = vm->swap_table[p];
		if(s_in == -1 && vm->storage_cnt < vm->STORAGE_SIZE){
			vm->swap_table[p] = vm->storage_cnt;
			s_in = vm->storage_cnt;
			vm->storage_cnt ++;
		}
		printf("s_in = %d\n", s_in);
		int data_out[32]; 
		for(int i = 0;i<vm->PAGESIZE; i++){
			data_out[i] = vm->storage[s_in*vm->PAGESIZE + i];
			vm->storage[s_in*vm->PAGESIZE+i] = vm->buffer[victim_f*vm->PAGESIZE + i];
		}
		f = victim_f;
		vm->invert_page_table[victim_f] = p;
		printf("f = %d\n", f);
		for(int i = 0; i < vm->PAGESIZE; i++){
			vm->buffer[f*vm->PAGESIZE+i] = data_out[i];
		}
		vm->swap_table[p] = -1;
		vm->swap_table[victim_p] = s_in;

		// printf("vm->swap_table[p] = %d\n", vm->swap_table[p]);
		// printf("vm->swap_table[victim_p] = %d\n", vm->swap_table[victim_p]);
	}
	out = vm->buffer[f*vm->PAGESIZE + d];
	vm->invert_page_table[f] = p;
	if(vm->LRU_top->page_number != p){
		update_LRU(vm, p);
	}
	if(addr % 32 == 0)
		printf("addr = %d from physical addr = {%d, f = %d}read out = %d\n",addr, f*vm->PAGESIZE + d, f, out);
  	return out; 
}

__device__ void vm_write(VirtualMemory *vm, u32 addr, uchar value) {
  	/* Complete vm_write function to write value into data buffer */
	int p = addr / vm->PAGESIZE;
	int d = addr % vm->PAGESIZE;
	int f = get_frameIdx(vm, p);				// find the corresponding frame

	if(f == -1){								// if the page number is not in the page table
		(*(vm->pagefault_num_ptr))++;
		int victim_p = vm->LRU_bottom->page_number;
		vm->LRU_bottom = vm->LRU_bottom->up;										// pop out
		int victim_f = get_frameIdx(vm, victim_p);
		int s_in = vm->swap_table[p];
		if(s_in == -1 && vm->storage_cnt < vm->STORAGE_SIZE){						// initialize the swap table entry
			vm->swap_table[p] = vm->storage_cnt;
			s_in = vm->storage_cnt;
			vm->storage_cnt ++;
		}
		printf("s_in = %d\n", s_in);
		int data_out[32];
		// store victime buffer to disk
		for(int i= 0; i < vm->PAGESIZE; i++){
			data_out[i] = vm->storage[s_in * vm->PAGESIZE + i];
			vm->storage[s_in * vm->PAGESIZE + i] = vm->buffer[victim_f*vm->PAGESIZE + i];
		}
		f = victim_f;
		vm->invert_page_table[victim_f] = p;
		// swap target disk to buffer
		for(int i = 0; i < vm->PAGESIZE; i++){
			vm->buffer[f*vm->PAGESIZE + i] = data_out[i];
		}
		vm->swap_table[p] = -1;
		vm->swap_table[victim_p] = s_in;
	}

	vm->buffer[f*vm->PAGESIZE + d] = value;
	vm->invert_page_table[f] = p;
	if(vm->LRU_top->page_number != p){
		update_LRU(vm, p);
	}
	if(addr % 32 == 0)
		printf("write buffer: logical address = {%d, p=%d} => physical address = {%d,f=%d} <- %d = %d\n ",addr,p, f*vm->PAGESIZE+d, f, vm->buffer[f*vm->PAGESIZE+d], value);
}	

__device__ void vm_snapshot(VirtualMemory *vm, uchar *results, int offset, int input_size) {
  	/* Complete snapshot function togther with vm_read to load elements from data
  	 * to result buffer */
	for(int i = 0; i<input_size; i++){
		int value = vm_read(vm, i+offset);
		results[i] = value;
	}
}                                                  
