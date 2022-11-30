# CSC3150 Project 4 File System

Tong Zhen 120090694

### 1. Program Environment

Linux Kernel Version

```shell
[120090694@node21 HM3]$ uname -r
3.10.0-862.el7.x86_64
```

cuda version

```Shell
[120090694@node21 HM3]$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Wed_Jun__8_16:49:14_PDT_2022
Cuda compilation tools, release 11.7, V11.7.99
Build cuda_11.7.r11.7/compiler.31442593_0
```

### 2. Program Structure Design

In this assignment, a range of volume array is used to simulate the  volume control block, file control block, and contents of the file. The program used cuda memory storage to simulate the concepts below. 

| Operating System Concept            | Size & Type Description                |
| ----------------------------------- | :------------------------------------- |
| Disk (1060KB) = VCB + FCB + Storage | `uchar[1085440]`                       |
| VCB (Super blocks) (4KB)            | `uchar [fs->SUPERBLOCK_SIZE]`          |
| FCB Blocks (32KB)                   | `uchar [fs->FCB_ENTRIES*fs->FCB_SIZE]` |
| FCB (32 Bytes)                      | `uchar [32]`                           |
| Storage blocks (1024KB)             | `uchar [1048576]`                      |
| Storage block (32 Bytes)            | `uchar [32]`                           |

#### Bit Map Structure

The VCB(Super blocks) is build in Bit Map format. A Bit Map is an array of number, whose every bit represent for a content storage condition. For example, `01001001` represents for 8 blocks of storage, the 1st, 3rd, 4th, 6th, 7th are empty storage block, and 2nd, 5th, 8th block are occupied. Each `uchar` number is a 8 bit representation for 8 blocks.

#### FCB Structure

The File Control Block (FCB), is a structure used to link logical file conception and physical data storage. This project design the FCB by 32 `uchar` number in this way:

 1. The 0-20 number is the file name, each number represents for a letter

 2. Data larger than 255 is represented by 2 `uchar` number, because the `uchar` is limited to max = 255
    $$
    data = uchar_1\times256 + uchar_2
    $$
    The create time, modify time, file size, map bit size, map bit location are all data larger than 255. They were recored by the 20-21, 22-23, 24-25, 26-27, 28-29 `uchar` number in one FCB block. 

```c
uchar *fcb=start_fcb + fcb_idx * fs->FCB_SIZE;
uchar *fcb_name = target_fcb;
uchar *create_time = target_fcb + 20;
uchar *modify_time = target_fcb + 22;
uchar *file_size = target_fcb + 24;
uchar *bit_size = target_fcb + 26;
uchar *bit_location = target_fcb + 28;

```

#### Function

##### fs_open(FileSystem *fs, char *s, int op)

![open](D:\Courses_2022_Fall\CSC3150\HM4\pics\open.jpg)

When the fcb is not found in the FCB region. We will allocate one spare fcb for it, and find the first empty bit  in the Bit Map for it.

##### fs_read(FileSystem *fs, uchar *output, u32 size, u32 fp)

With the input fp as a FCB index, we can find the FCB pointer in the volume. The FCB pointer will give the location of the data and how many blocks it last.

##### fs_write(FileSystem *fs, uchar* input, u32 size, u32 fp)

![write](D:\Courses_2022_Fall\CSC3150\HM4\pics\write.jpg)

When writing a file, the current allocated size may not fit. If more storage were given, set the extra bit map to zero. When the current allocated storage size is less than the required size, we need first set the allocated storage to zero.

##### find_first_fit(FileSystem *fs, int bits_num)

As we mentioned above, the suitable bit location with a certain bit size should be found by first fit (first fit has similar performance with best fit algorithm).  We search the Bit Map, and find the first contiguous given size bit location.

![firstfit](D:\Courses_2022_Fall\CSC3150\HM4\pics\firstfit.jpg)



##### compact_storage(FileSystem *fs)

![compact](D:\Courses_2022_Fall\CSC3150\HM4\pics\compact.jpg)

When the process encounters no more suitable bit in the Bit Map for writing data, a compact occur would occur. Its motivation is to squeeze the occupied bit to the front of the VCB tightly. Things worth to be mention:

 	1. The FCB corresponding to the start-bit location is search in FCB region with O(N) time.
 	2. The physical storage of the file should be moved to the new indicated region.

##### fs_gsys(FileSystem *fs, int op)

The `op` can be `LS_D`(modified time), or `LS_S`(file size). The program first scan all the fcbs in the FCB reigion, add them into an array, and do bubble sort in descending order.

#####  fs_gsys(FileSystem *fs, int op, char *s)

The `op` is `RM` here, do remove. With given file name, the file system find the corresponding fcb. The bit map meta data is set to zero, the storage of the fcb is set to zero, and the FCB it self is freed.

### Run

Two way to run the program

```shell
make
./test
```

```shell
./slurm.sh
./test
```

case 1

<img src="C:\Users\surface\AppData\Roaming\Typora\typora-user-images\image-20221128104236966.png" alt="image-20221128104236966" style="zoom:50%;" />

case 2

<img src="C:\Users\surface\AppData\Roaming\Typora\typora-user-images\image-20221128110513795.png" alt="image-20221128110513795" style="zoom:50%;" />

case 3

<img src="C:\Users\surface\AppData\Roaming\Typora\typora-user-images\image-20221128110931027.png" alt="image-20221128110931027" style="zoom:50%;" /><img src="C:\Users\surface\AppData\Roaming\Typora\typora-user-images\image-20221128111006762.png" alt="image-20221128111006762" style="zoom:50%;" /><img src="C:\Users\surface\AppData\Roaming\Typora\typora-user-images\image-20221128111115585.png" alt="image-20221128111115585" style="zoom:50%;" /><img src="C:\Users\surface\AppData\Roaming\Typora\typora-user-images\image-20221128111221402.png" alt="image-20221128111221402" style="zoom:50%;" />

Compare part of the written and read data with the input.

```shell
cmp data.bin snapshot.bin -i 1000 1000
```

<img src="C:\Users\surface\AppData\Roaming\Typora\typora-user-images\image-20221126112334972.png" alt="image-20221126112334972" style="zoom:67%;" />

case 4





<img src="C:\Users\surface\AppData\Roaming\Typora\typora-user-images\image-20221128113624315.png" alt="image-20221128113624315" style="zoom:50%;" /><img src="C:\Users\surface\AppData\Roaming\Typora\typora-user-images\image-20221128113711842.png" alt="image-20221128113711842" style="zoom:50%;" />

Compare the output `snapshot.bin` with the input.

```
[120090694@node21 HM4]$ cmp snapshot.bin data.bin 
```

<img src="C:\Users\surface\AppData\Roaming\Typora\typora-user-images\image-20221127211938923.png" alt="image-20221127211938923" style="zoom:73%;" />

### Bonus

#### Directory Structure:

The directory concept is similar to file concept, except that it doesn't need to store data. The name, create time and modify time meta data location in FCB are the same. The difference is as follows:

1. The `son_fcb_idx` record the first son fcb index in the FCB region.
2. the son number is the length of the singly linked list of sons (max length 50)
3. `father` is the fcb index of the directory's parent directory 
4. `next_bro` is the next sibling's fcb index.
5. The File System has a new variable called `cur_DIRfcb_idx` a `int` indicate the current directory fcb index.

```c
				uchar *create_time = dir_fcb + 20;
				uchar *modify_time = dir_fcb + 22;
				gtime++;
				*create_time = gtime/256;
				*(create_time + 1) = gtime%256;
				*modify_time = gtime/256;
				*(modify_time + 1) = gtime%256;

				uchar *son_fcb_idx = dir_fcb + 24;
				uchar *son_number = dir_fcb + 26;
				uchar *father = dir_fcb + 28;
				*father = fs->cur_DIRfcb_idx/256;
				*(father +1) = fs->cur_DIRfcb_idx%256;
				uchar *next_bro = dir_fcb + 30;

```

The File System structure with directory tree is:

![dir](D:\Courses_2022_Fall\CSC3150\HM4\pics\dir.jpg)

The affiliation of the tree structure is constructed by singly linked list. The sort and remove operation are based on this structure.

#### Functions

##### MKDIR

Under the current directory , find a empty fcb and initialize the father of the new directory-fcc as `cur_DIRfcb_idx`.

##### CD

Set the `cur_DIRfcb_idx` to the given fcb index

##### CD_P

Find the father of the `cur_DIRfcb_idx` fcb and set the `cur_DIRfcb_idx` to the father fcb index

##### PWD

find the father of the directory-fcb pointer until its fcb index == 0 (the /root fcb when the File System initialed). 

![dir](C:\Users\surface\Downloads\dir.jpg)

##### RM_RF

Remove the given directory and all its subdirectories and files recursively. Instead of really do it recursively, the program builds a queue to remove with while loop.

![rmrf](D:\Courses_2022_Fall\CSC3150\HM4\pics\rmrf.jpg)

#### Run

Two way to run the bonus

```shell
make
./test
```

```shell
./slurm.sh
./test
```

#### Output

![image-20221129170335641](C:\Users\surface\AppData\Roaming\Typora\typora-user-images\image-20221129170335641.png)
