#include "virtual_memory.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

__device__ void init_invert_page_table(VirtualMemory *vm) {
	
  	for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
  	  	vm->invert_page_table[i] = 2;
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

	// for(int i = 0; i < 100; i++){
	// 	printf("%d ", vm->invert_page_table[i + vm->PAGE_ENTRIES]);
	// }
	
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

__device__ void update_LRU(VirtualMemory *vm, int page_num){	// update the stack 

	struct memory_item* temp = vm->LRU_bottom;
	struct memory_item* target = (struct memory_item *)malloc(sizeof(memory_item));	
	if(page_num == 90676)
		printf("Attention\n");

	if(temp->page_number == -1){								// if the first is unused
		temp->page_number = page_num;
		vm->LRU_top->page_number = page_num;
		return;
	}
	if(temp->page_number == page_num && temp->up == NULL){		// if the only element is the pagenum
		return;
	}
	if(vm->LRU_top->page_number == page_num){					// if the top element is the pagenum
		return;
	}
	while(temp->up != NULL){

		if(temp->up->page_number == page_num){
			target = temp->up;
			temp->up = target->up;
			temp = temp->up;

			while(temp->up != NULL){
				temp = temp->up;
			}
			temp->up = target;
			vm->LRU_top = target;
			// printLRUStack(vm);
			return;

		}
		temp = temp->up;
	}	
	target->page_number = page_num;
	temp->up = target;
	vm->LRU_top = target;
	return;
}

__device__ void delete_LRU(VirtualMemory * vm, int page_num){
	if(vm->LRU_bottom == NULL){
		// printf("LRU: empty\n");
		return;
	}
	struct memory_item *temp = vm->LRU_bottom;
	if(vm->LRU_bottom->page_number == page_num && vm->LRU_bottom->up != NULL){
		// printf("LRU: delete the first one\n");
		vm->LRU_bottom = vm->LRU_bottom->up;
		return;
	}
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

__device__ uchar vm_read(VirtualMemory *vm, u32 addr) {
  	/* Complate vm_read function to read single element from data buffer */

	int p = addr / vm->PAGESIZE;
	int d = addr % vm->PAGESIZE;
	int f = vm->invert_page_table[p];
	uchar out;
	update_LRU(vm, p);

	if(vm->invert_page_table[p+vm->PAGE_ENTRIES] == 0){
		// printf("[read] invalid: page = %d state=%d\n", p, vm->invert_page_table[p+vm->PAGE_ENTRIES]);

		vm->pagefault_num_ptr++;
		// perform page replacement
		// printf(" perform page replacement\n");
		int victim_p = vm->LRU_bottom->page_number;
		// printf("victim_p = %d\n", victim_p);
		// int victim_f = vm->invert_page_table[victim_p];
		int victim_f = victim_p %(vm->PHYSICAL_MEM_SIZE/vm->PAGESIZE);
		// printf("[read]step 1: swap out victim page\n");
		for(int i = 0; i < vm->PAGESIZE; i++){
			vm->storage[victim_f*vm->PAGESIZE+i] = vm->buffer[victim_p*vm->PAGESIZE+i];
			// printf("%d ", vm->storage[victim_f*vm->PAGESIZE+i]);
		}
		// step 2: change to invalid
		vm->invert_page_table[victim_p] = 0;
		delete_LRU(vm, victim_p);
		// printf("[read]step 3: swap derised page in\n");
		for (int i = 0; i < vm->PAGESIZE; i++){
			vm->buffer[f*vm->PAGESIZE+i] = vm->storage[f*vm->PAGESIZE+i];
			printf("%d ", vm->buffer[f*vm->PAGESIZE+i]);
		}
		// read the value
		out = vm->buffer[f*vm->PAGESIZE+d];
		// step 4: reset page table as valid for new page
		vm->invert_page_table[p+vm->PAGE_ENTRIES] = 1;
		delete_LRU(vm, victim_p);
	}
	else{
		// printf("valid\n");
		out = vm->buffer[f*vm->PAGESIZE + d];
	}
	printf("read out = %d\n", out);
  	return out; 
}

__device__ void vm_write(VirtualMemory *vm, u32 addr, uchar value) {
  	/* Complete vm_write function to write value into data buffer */
	int p = addr / vm->PAGESIZE;
	int d = addr % vm->PAGESIZE;
	int f = p %(vm->PHYSICAL_MEM_SIZE/vm->PAGESIZE);
	// periodic allocate the f to the p in page table
	// printf("p:%d, d:%d, f:%d\n", p, d, f);
	// printf("addr = %d, vallue = %d\n", addr, value);

	
	update_LRU(vm, p);
	// if the page is unused
	if(vm->invert_page_table[p] == 2){
		// printf("[write]unused:page = %d state=%d\n", p, vm->invert_page_table[p+vm->PAGE_ENTRIES]);
		vm->buffer[f*vm->PAGESIZE+d] = value;
		vm->invert_page_table[p] = 1;
	}
	// if the page entry is invalid
	else if(vm->invert_page_table[p] == 0){
		// printf("[write] invalid: page = %d state=%d\n", p, vm->invert_page_table[p+vm->PAGE_ENTRIES]);
		vm->pagefault_num_ptr++;
		// perform page replacement
		int victim_p = vm->LRU_bottom->page_number;
		// int victim_f = vm->invert_page_table[victim_p];
		int victim_f = victim_p %(vm->PHYSICAL_MEM_SIZE/vm->PAGESIZE);

		// printf("victim_p = %d, victim_f = %d\n", victim_p, victim_f);
		// printf("[write]step 1: swap out victim page\n");
		for(int i = 0; i < vm->PAGESIZE; i++){
			vm->storage[victim_p*vm->PAGESIZE+i] = vm->buffer[victim_f*vm->PAGESIZE+i];
			// printf("%d ", vm->storage[victim_f*vm->PAGESIZE+i]);
		}
		// step 2: change to invalid
		vm->invert_page_table[victim_p] = 0;
		// printf("[write]step 3: swap derised page in");
		for (int i = 0; i < vm->PAGESIZE; i++){
			vm->buffer[f*vm->PAGESIZE+i] = vm->storage[p*vm->PAGESIZE +i];
			// printf("%d ", vm->buffer[f*vm->PAGESIZE+i]);
		}
		// write in new value
		vm->buffer[f*vm->PAGESIZE+d] = value;
		// step 4: reset page table as valid for new page
		vm->invert_page_table[p] = 1;
		delete_LRU(vm, victim_p);
		// printLRUStack(vm);

	}
	else{
		// printf("valid: %d %d = %d\n", vm->invert_page_table[p + vm->PAGE_ENTRIES],vm->invert_page_table[p], f);
		vm->buffer[f*vm->PAGESIZE + d] = value;
	}
	printf("write buffer: logical address = {%d, p=%d} => physical address = {%d,f=%d} <- %d = %d\n ", 
				addr,p, f*vm->PAGESIZE+d, f, vm->buffer[f*vm->PAGESIZE+d], value);
}

__device__ void vm_snapshot(VirtualMemory *vm, uchar *results, int offset, int input_size) {
  	/* Complete snapshot function togther with vm_read to load elements from data
  	 * to result buffer */
}
