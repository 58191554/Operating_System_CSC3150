#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__device__ __managed__ u32 gtime = 0;


__device__ void fs_init(FileSystem *fs, uchar *volume, int SUPERBLOCK_SIZE,
							int FCB_SIZE, int FCB_ENTRIES, int VOLUME_SIZE,
							int STORAGE_BLOCK_SIZE, int MAX_FILENAME_SIZE, 
							int MAX_FILE_NUM, int MAX_FILE_SIZE, int FILE_BASE_ADDRESS)
{
  	// init variables
  	fs->volume = volume;

  	// init constants
  	fs->SUPERBLOCK_SIZE = SUPERBLOCK_SIZE;
  	fs->FCB_SIZE = FCB_SIZE;
  	fs->FCB_ENTRIES = FCB_ENTRIES;
  	fs->STORAGE_SIZE = VOLUME_SIZE;
  	fs->STORAGE_BLOCK_SIZE = STORAGE_BLOCK_SIZE;
  	fs->MAX_FILENAME_SIZE = MAX_FILENAME_SIZE;
  	fs->MAX_FILE_NUM = MAX_FILE_NUM;
  	fs->MAX_FILE_SIZE = MAX_FILE_SIZE;
  	fs->FILE_BASE_ADDRESS = FILE_BASE_ADDRESS;

}

__device__ bool check_name(uchar *fcb, char *s){
	// check byte by byte of the two name
	while(*fcb!='\0' && *s !='\0'){
		if(*fcb == *s){
			fcb ++;
			s ++;
		}
		else
			break;
	}
	if(*fcb == '\0' && *s == '\0')
		return true;
	else
		return false;
}

__device__ u32 find_empty_fcb(FileSystem *fs){
	// find a empty space in FILE CONTROL BLOCK

	for(u32 i = 0; i<fs->FCB_ENTRIES;i++){
		uchar *fcb_i = fs->volume+fs->SUPERBLOCK_SIZE+i*fs->FCB_SIZE;
		if(*fcb_i == 0)
			return i;
	}
	// if no empty fcb, return a number out of the FILE CONTROL BLOCK
	return (u32)(-1);
}

__device__ u32 find_empty_bit(FileSystem *fs){
	// allocate a empty or semi-empty bit in the SUPER BLOCK
	uchar *bit_map = fs->volume;
	for(int i = 0; i < fs->SUPERBLOCK_SIZE; i++){
		// printf("bit[%d] = %d\n",i ,*(fs->volume+i));
		if(*(bit_map+i) == 0){
			return i;
		}
	}
	printf("no emtpy bit in SUPER BLOCK");
}

__device__ u32 fs_open(FileSystem *fs, char *s, int op)
{
	/* Implement open operation here */
	// input the file name *s -> return the fcb_index int FCBs
	printf("\n[open file]:%s\n", s);
	u32 fcb_idx;
	uchar *start_fcb = fs->volume + fs->SUPERBLOCK_SIZE;
	bool file_exist = false;

	// find the file name in FCB
	for(int i = 0; i < fs->FCB_ENTRIES; i++){
		uchar *fcb = start_fcb + i * fs->FCB_SIZE;
		// printf("FCB[%d]:%d\n",i,  *fcb);
		if(check_name(fcb, s)){
			printf("find file\n");
			file_exist = true;
			fcb_idx = i;
		}
	}

	if(file_exist){
		return fcb_idx;
	}
	else{
		// find empty space for writting a new file
		if(op == G_WRITE){
			// search for the empty fcb
			fcb_idx = find_empty_fcb(fs);
			printf("empty fcb = %d\n", fcb_idx);

			uchar *target_fcb=start_fcb + fcb_idx * fs->FCB_SIZE;
			uchar *target_name = target_fcb;
			uchar *target_create_time = target_fcb + 20;
			uchar *target_modify_time = target_fcb + 22;
			uchar *target_file_size = target_fcb + 24;
			uchar *target_bit_size = target_fcb + 26;
			uchar *target_bit_location = target_fcb + 28;
			uchar *target_fcb_idx = target_fcb + 30;

			// write file name
			for(char *c = s; *c!='\0'; c++){
				*target_name = *c;
				target_name ++;
			}
			*target_name = '\0'; 
			gtime++;
			*target_create_time = gtime;
			*target_modify_time = gtime;
			*target_bit_size = 1;
			*target_file_size = 0;
			*target_fcb_idx = fcb_idx;
			// find empty or semi-empty bit
			u32 empty_bit = find_empty_bit(fs);
			printf("empty_bit = %d\n", empty_bit);
			*target_bit_location = empty_bit;

			// occupy the empty bit
			*(fs->volume+*target_bit_location) = 1;
			return fcb_idx;
		}
		else if(op == G_READ){
			printf("No readable file named %c", *s);
		}
	}
}

__device__ void fs_read(FileSystem *fs, uchar *output, u32 size, u32 fp)
{
	/* Implement read operation here */
	uchar *target_fcb = fs->volume + fs->SUPERBLOCK_SIZE + fp*fs->FCB_SIZE;
	// uchar *target_fcb_modify_time = target_fcb + 22;
	// uchar *bit_length = target_fcb + 26;
	uchar *bit_location = target_fcb + 28;

	uchar *reader = fs->volume + 
					fs->SUPERBLOCK_SIZE + 
					(fs->FCB_ENTRIES*fs->FCB_SIZE) + 
					fs->STORAGE_BLOCK_SIZE*(*bit_location);
	
	printf("[read]\n");
	for(int i = 0; i < size; i++){
		*output = *reader;
		printf("%d ", *output);
		reader ++;
		output++;
	}
	printf("\n");
}

__device__ u32 find_fcb_from_bit(FileSystem *fs, int bit_num){
	// given the bit_num, return the corresponding fcb
	uchar *start_fcb = fs->volume + fs->SUPERBLOCK_SIZE;
	for(int i = 0; i<fs->FCB_ENTRIES; i++){
		uchar *target_bit_location = start_fcb + i*fs->FCB_SIZE + 28;
		if(*(target_bit_location) == bit_num){
			printf("find the fcb of the bit\n");
			return i;
		}
	}
	printf("ERROR: NO FCB OF THE BIT\n");
}

__device__ void compact_storage(FileSystem *fs){
	printf("\n[compact]\n");
	// set the copy storage array as the MAX_FILE_SIZE
	uchar copy_storage[1048576];
	// copy the volume to another disk
	uchar *bit_ptr = fs->volume;
	int bit_step = 0;

	while(bit_step < fs->SUPERBLOCK_SIZE){
		if(*bit_ptr == 0){
			// if a hole
			uchar fp = find_fcb_from_bit(fs, bit_step);
		}
	}
}

__device__ u32 find_first_fit(FileSystem *fs, int block_num){
	// if the file length is larger than the current length, find the first fit location.
	printf("[find_first_fit], block number = %d\n", block_num);
	uchar *start_bit = fs->volume;
	int step = 0;
	while(step < fs->SUPERBLOCK_SIZE){
		// printf("finding step = %d\n", step);
		int i;
		for(i = 0; i < block_num; i++){
			uchar *cur_bit = start_bit + step;
			// printf("length = %d\n", i);
			if(*cur_bit == 1){
				// printf("[bit %d]: Not good\n", step);
				step ++;
				break;
			}
			step ++;
		}
		if(i == block_num){
			// step++;
			printf("find! first fit bit = %d\n", step-block_num);
			return step-block_num;
		}
	}
	// no feasible storage space
	printf("NO FEASIBLE STORAGE, DO COMPACT\n");
	// compact_storage(fs);
}

__device__ u32 fs_write(FileSystem *fs, uchar* input, u32 size, u32 fp)
{
	/* Implement write operation here */
	uchar *target_fcb = fs->volume + fs->SUPERBLOCK_SIZE + fp*fs->FCB_SIZE;
	uchar *target_modify_time = target_fcb + 22;
	uchar *target_file_size = target_fcb + 24;
	uchar *target_bit_size = target_fcb + 26;
	uchar *target_bit_location = target_fcb + 28;
	// uchar *target_fcb_idx = target_fcb + 30;

	uchar *writer = fs->volume + 
					fs->SUPERBLOCK_SIZE + 
					(fs->FCB_ENTRIES*fs->FCB_SIZE) + 
					fs->STORAGE_BLOCK_SIZE*(*target_bit_location);

	// update file size
	*target_file_size = size;

	// find suitable physical location
	int need_bit_size = ceil((float)(size)/(float)(fs->STORAGE_BLOCK_SIZE));
	if((*target_bit_size)<need_bit_size){
		// if we need more space to store the data
		// set current bit map to 0, free the space in the bit-map.
		for(uchar i = 0; i < (*target_bit_size); i++){
			uchar *bit = fs->volume + *target_bit_location + i;
			*bit = 0;
		} 

		// reset the data in the STORAGE
		uchar *start_block = fs->volume + 
							fs->SUPERBLOCK_SIZE+
							(fs->FCB_ENTRIES*fs->FCB_SIZE) + 
							(*target_bit_location)*(fs->STORAGE_BLOCK_SIZE);
		
		for(uchar b = 0; b < *target_bit_size; b++){
			uchar *block = start_block + b*(fs->STORAGE_BLOCK_SIZE);
			for(int i = 0; i < fs->STORAGE_BLOCK_SIZE; i++){
				*block = '\0';
				block ++;
			}
		}
		
		// find the first fit bit
		int first_fit = find_first_fit(fs, need_bit_size);
		// printf("bit_location = %d\n",first_fit);

		// transfer int to uchar
		*target_bit_location = first_fit/256;
		*(target_bit_location+1) = first_fit%256;

		printf("first_fit = %d\n", (*target_bit_location)*256+*(target_bit_location+1));

		// set bit_length to need_bit_size
		*target_bit_size = (uchar)need_bit_size;
		// occupy bit map
		uchar *bit = fs->volume + first_fit;
		for(int i = 0; i < *target_bit_size; i++){
			*bit = 1;
			printf("[occupy bit %d] = %d\n", (first_fit+i), *bit);
			bit++;
		}	
	}
	else if((*target_bit_size)< need_bit_size){
		// if the need_bit_size < current bit size, we set some bit to 0 to free some space in bit map
		uchar *free_bit = fs->volume + *target_bit_location + need_bit_size;
		for(int i = 0; i < (*target_bit_size-need_bit_size); i++){
			*free_bit = 0;
			free_bit ++;
		}
	}

	// write data into STORAGE
	printf("[write data]:size = %d\n", size);
	for(int i = 0; i < size; i++){
		// printf("%d", *input);
		*writer = *input;
		writer ++;
		input++;
	}
	// printf("\n");

	// modify time update
	gtime ++;
	*target_modify_time = gtime;

	return fp;
}

__device__ void fs_gsys(FileSystem *fs, int op)
{
	/* Implement LS_D and LS_S operation here */
	// get the fcb number and fcb array
	uchar *fcb_arr[1024];
	int fcb_num = 0;
	uchar *fcb_ptr = fs->volume + fs->SUPERBLOCK_SIZE;
	for(int i = 0; i < fs->FCB_ENTRIES; i++){
		// printf("fcb_arr[%d] = %s\n",i, fcb_ptr);
		if(*fcb_ptr != '\0'){
			fcb_arr[fcb_num] = fcb_ptr;
			fcb_num ++;
		}
		fcb_ptr+=fs->FCB_SIZE;
	}

	if(op == LS_D){
		printf("\n===sort by modified time===\n");
		// // use bubble sort
		for(int i = 0; i < fcb_num-1; i++){
			uchar *m_time_latest = fcb_arr[i] + 22;
			int max_idx = i;
			for(int j = i+1; j < fcb_num; j++){
				uchar *m_time_j = fcb_arr[j] + 22;
				if(*m_time_j>*m_time_latest){
					m_time_latest = m_time_j;
					max_idx = j;
				}
			}
			uchar *fcb1 = fcb_arr[i];
			uchar *fcb2 = fcb_arr[max_idx];
			fcb_arr[i] = fcb2;
			fcb_arr[max_idx] = fcb1; 
		}

		// uchar *fcb_start = fs->volume+fs->SUPERBLOCK_SIZE;
		for(int i = 0; i<fcb_num; i++){
      		uchar *fcb_cur = fcb_arr[i];
      		printf("%s\n",fcb_cur);		
		}
	}
	else if(op == LS_S){
		printf("\n===sort by file size===\n");
		for(int i = 0; i < fcb_num-1; i++){
			uchar *max_len = fcb_arr[i] + 24;
			int max_idx = i;
			for(int j = i+1; j < fcb_num; j++){
				uchar *len_j = fcb_arr[j] + 24;
				if(*len_j>*max_len){
					max_len = len_j;
					max_idx = j;
				}
			}
			uchar *fcb1 = fcb_arr[i];
			uchar *fcb2 = fcb_arr[max_idx];
			fcb_arr[i] = fcb2;
			fcb_arr[max_idx] = fcb1; 
		}

		// uchar *fcb_start = fs->volume+fs->SUPERBLOCK_SIZE;
		for(int i = 0; i<fcb_num; i++){
      		uchar *fcb_cur = fcb_arr[i];
			uchar *len = (uchar*)(fcb_cur + 24);
      		printf("%s %d\n",fcb_cur, *len);		
		}

	}
}

__device__ void fs_gsys(FileSystem *fs, int op, char *s)
{
	/* Implement rm operation here */
	printf("\n[Remove]:%s\n", s);
	int fcb_idx;
	uchar *start_fcb = fs->volume+fs->SUPERBLOCK_SIZE;
	bool file_exist = false;

	// find the file name in FCBs
	for(int i = 0; i<fs->FCB_ENTRIES; i++){
		uchar *fcb = start_fcb + i*fs->FCB_SIZE;
		if(check_name(fcb, s)){
			printf("[find file]: %s\n", fcb);
			file_exist = true;
			fcb_idx = i;
		}
	}

	if(file_exist == false){
		printf("ERROR: FILE DOES NOT EXIST!\n");
		return;
	}

	uchar *target_fcb = start_fcb + fcb_idx*fs->FCB_SIZE;
	// uchar *target_modify_time = target_fcb + 22;
	// uchar *target_file_size = target_fcb + 24;
	uchar *target_bit_size = target_fcb + 26;
	uchar *target_bit_location = target_fcb + 28;
	// uchar *target_fcb_idx = target_fcb + 30;

	// clean the bit map
	uchar *rm_bit = fs->volume + *target_bit_location;
	for(int i = 0; i < *target_bit_size; i++){
		*rm_bit = 0;
		rm_bit ++;
	}

	// clean the STORAGE
	uchar *rm_storage = fs->volume + fs->SUPERBLOCK_SIZE + 
						(fs->FCB_ENTRIES*fs->FCB_SIZE);
	for(int b = 0; b < *target_bit_size; b++){
		for(int i = 0; i < fs->STORAGE_BLOCK_SIZE; i++){
			*rm_storage = 0;
			rm_storage ++;
		}
	}

	// set RM fcb to '\0'
	for(int i = 0; i < fs->FCB_SIZE; i++){
		if(*target_fcb == '\0')
			break;
		// printf("[rm fcb] = %s\n", target_fcb);
		*target_fcb = 0; 
		target_fcb ++;
	}
}

