# CSC3150 Assignment 3
## Environment
![uname](HM3pics\uname.png)
---
## Task description
In Assignment 3, we are required to simulate a mechanism of virtual memory via GPU's memory.\
Two memories (global memory and shared memory) related to our project. 
- Global memory **(like disk storage)**
    - Typically implemented in DRAM
    - High access latency: 400-800 cycles
- Shared memory **(like physical memory)**
    - Extremely fast
    - Configurable cache
    - Memory size is small (16 or 48 KB)

![](HM3pics\MemoryStructure.png)
In CUDA, the function executed on GPU that defined by programmer, is called kernel
function. \
内核函数将不再受可用的共享内存量的限制。用户将能够为一个非常大的虚拟地址空间编写内核函数，从而简化编程任务。\
Implement a paging system with **swapping** where the thread access data in shared memory and retrieves data from global memory (secondary memory). \
We **only implement the data swap when page fault occurs** (not in the instruction level). \
在作业中我们使用 cuda memory 是这样来模拟逻辑内存（虚拟内存），物理内存，磁盘空间的
![](HM3pics\cudaMemorySimulation.png)

## Motivation
In the users view, the logical memory is large. However, the pysical memory is small and limited. We are going to translate the logical memory from the pysical memory. When we the user run out of the physical memory, we need to swap the data with the disk. 
- We need to implement the 2 things:
    - page table
    - LRU algorithm

### Cuda Background

Memory declared by `__decive__` will be store in global memory, the large one we will used to simulate the disk.The `__share__`memory what we need to simulate as the pysical memory.

A cuda program will use multiple threads. Each block contains a group of threads, and each gird contains a group of blocks. 
The cuda program paradigm is that each thread will execute the same instruction but on separate piece of data.
___
### Detail Implementation
#### Load binary file
#### Run `mykernel`
- vm_init
    - 建立一个 `VirtualMemory`的指针变量， 并初始化他
    - data
    - storage
    - vm->buffer = buffer;
    - vm->storage = storage;
    - vm->invert_page_table = invert_page_table;
    - vm->pagefault_num_ptr = pagefault_num_ptr;
    - 
    - // init constants
    - vm->PAGESIZE = PAGESIZE;
    - vm->INVERT_PAGE_TABLE_SIZE = INVERT_PAGE_TABLE_SIZE;
    - vm->PHYSICAL_MEM_SIZE = PHYSICAL_MEM_SIZE;
    - vm->STORAGE_SIZE = STORAGE_SIZE;
    - vm->PAGE_ENTRIES = PAGE_ENTRIES; 
    \
    - user_program
        - write
        - read
