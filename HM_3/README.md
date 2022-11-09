# CSC3150 Project 3 Report
Tong Zhen 120090694
## 1. Program Environment
Linux kernel version
```shell
[120090694@node21 HM3]$ uname -r
3.10.0-862.el7.x86_64
```
cuda version
```shell
[120090694@node21 HM3]$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Wed_Jun__8_16:49:14_PDT_2022
Cuda compilation tools, release 11.7, V11.7.99
Build cuda_11.7.r11.7/compiler.31442593_0 
```

## 2. Program Design
|Operation System Concept | Size & Type Description|
|----|----|
|User Logical Memory(160KB)|`STORAGE_SIZE + PHYSICAL_MEM_SIZE` (int[163,840])|
|Inverted Page Table(16KB)|`vm->invert_page_table` (int [2048])|
|Main Memory / Physical Memory (32KB)|`vm->buffer` (int [32768])|
|Disk / Secondary Memory (128KB) |`vm->storage` (int [131072])|
|LRU stack|`vm->LRU_bottom` to `vm->LRU_top`(1024 node)|
|Swap table(5KB)|`vm->swap_table`(int[5120])|

The program used cuda memory storage to simulate the concepts above
My design flow chart is as follows:
![](HM3pics\flowChart.drawio.png)
1. First, the `vm_read()` and `vm_read()` both visit a virtual address, and the page should be found. The logical page number is the vm_addr divided by page size 32. The offset is the vm_addr mod the page size 32.
2. Second, find the physical page number by the logical page number $p$ in inverted page table
    a. If we find the physical page number successfully in the inverted page tabel, which means the data we want to visit is in the physical memory (buffer), go to step 4

    b. If we didn't find the physical page in the inverted page table, which means the data we want to visit is in the storage, do swap
3. Swap: We need to get the vitim frame in the logical memory(buffer), say `vicitim_p` that is popped from the LRU stack. Search the physical memory, say`vicitim_f` in the inverted page table. Search the storage location in the swap table, say`s_in`. If the storage location we get is -1, which means it un-used, we assign it to a number less than storage maximum to activate. Remove the buffer data to the storage and remove the storage data to the buffer. Assign the victim item in swap table to -1 to inactivate, and assign the visit page item to `s_in` in swap table

4. Do write new data over the old data or read out the data in the target position, then update the LRU stack

## 3. Function Information
`init_invert_page_table` 
The function uses 1024 integer to storage the page number cooresponding to the physical frame number. Initialize them by -1
`init_swap_table`
The function uses 1024+2096 integer to storage the storage location information for the virtual page, Initialize them by -1
`showLRU` 
This is a function used to type the LRU stack information, used for debug.
`update_LRU`
When a new item comes in a LRU stack, it will find the previous position of the item and remove it to the top of the LRU stack
When a victim is needed because of swapping, pop the bottom of the LRU stack 
`get_frameIdx`
This function is used to find the physical frame page in the inverted page table and return the corresponding index.
`vm_read`
This function will read a data by the virtual address given. The operation is described above.
`vm_write` 
Same as vm_read, do the visit operation and write in the data.
`vm_snapshot`
This function is used to read all the data in the physical memory and the storage, output it as a array. The offset argument is used to do the circular move in the output array.


## 4. Run 
use make file
```shell
# build
[120090694@node21 HM3]$ make
```
use nvcc directly build
```shell
# buid
[120090694@node21 HM3]$ nvcc --relocatable-device-code=true main.cu user_program.cu virtual_memory.cu -o test
```
run
```shell
# run
/[120090694@node21 HM3]$ ./test
```
compare
```shell
[120090694@node21 HM3_bonus]$ cmp data.bin snapshot.bin
```

## 5. Output information and explanation
Task 1
![](HM3pics\task1.png)
Specifically, the total pagefault number for vm_write() is 4096, and 1 for vm_read(), and 4096 for vm_snapshot().
Explanation: Page fault occurs when the targeted page is not found in the page table. When the logical address is transmitted to the function, the page number is obtained by certain calculation. The program will search the page number in the page table. As long as it does not find the page number, a page fault will occur.


Task 2
![](HM3pics\task2.png)
The total page fault for the first loop is 4096, and the secodn for loop is 1023, and the snapshot read take 4096 page fault.

## 6. Bonus
In bonus version 1,
 the general visit step is similar to the flow chart above. Here declear the difference.
1. First, in `main.cu` we need to launch 4 threads:
    ```cuda
    mykernel<<<1, 4, INVERT_PAGE_TABLE_SIZE>>>(input_size);
    ```
2. Second, in the `user_program.cu` the task is done concurrently, which need to divide the give each thread its own logical address to visit.It is the same when we need to read all the data and do the snapshot
    ```cuda
        for(int i = 0; i < input_size/4; i++){
            int addr = i + thread_id*(input_size/4);
    ```

3. Third, the `invert_page_table` adds 1024 integer for the `thread_id`. They are firstly initialized as -1.
4. Fourth, when finding index of the frame of a given virtual page number in the inverted page table, we need to confirm the thread_id also. 
5. Fifth, the `vm->storage_cnt`(a int) will add 1, whenever it meets the unused swap table elements before it meets the storage maximum. However, each thread will add 1 indepentantly, which the actual `vm->storage_cnt` is uncertain. Therefore, the `vm->storage_cnt` is divided to 4 parts for the the threads to add without mistake.

**Run**
The same way as the normal task, **CHANGE THE TEST IN USER_PROGRAM.CU**


**Output**
Run the write all data, read all data, and do snapshot
```cuda
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

    for(int i = 0; i < input_size/4 ; i ++){

        int addr = i + thread_id*(input_size/4);
        if(thread_id == 0)
            vm_read(vm, addr, thread_id);
        __syncthreads();
        if(thread_id == 1)
            vm_read(vm, addr, thread_id);
        __syncthreads();
        if(thread_id == 2)
            vm_read(vm, addr, thread_id);
        __syncthreads();
        if(thread_id == 3)
            vm_read(vm, addr, thread_id);
        __syncthreads();
    }

    vm_snapshot(vm, results, 0, input_size, thread_id);
```
![](HM3pics\bonus_write_read_snap.png)
Run the task 1(Change user_program.cu to support the multi-threads)
![](HM3pics\task1bonus.png)
Run task 2, the offset write process need to do the index division for the 4 thread.
![](HM3pics\task2bonus.png)
