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

__device__ int find_empty_bit(FileSystem *fs){
	// allocate a empty or semi-empty bit in the SUPER BLOCK
	uchar *bit_map = fs->volume;
	for(int i = 0; i < fs->SUPERBLOCK_SIZE; i++){
		uchar *bits = bit_map+i;
		for(int j = 0; j < 8; j++){
			int bit = (*bits >> j)%2;
			if(bit == 0){
				return i*8+j;
			}
		}
	}
	printf("no emtpy bit in SUPER BLOCK");
}

__device__ u32 fs_open(FileSystem *fs, char *s, int op)
{
	/* Implement open operation here */
	// input the file name *s -> return the fcb_index int FCBs
	// printf("\n[open file]:%s\n", s);
	u32 fcb_idx;
	uchar *start_fcb = fs->volume + fs->SUPERBLOCK_SIZE;
	bool file_exist = false;

	// find the file name in FCB
	for(int i = 0; i < fs->FCB_ENTRIES; i++){
		uchar *fcb = start_fcb + i * fs->FCB_SIZE;
		if(check_name(fcb, s)){
			file_exist = true;
			fcb_idx = i;
		}
	}

	if(file_exist){
		// printf("fcb = %d\n", fcb_idx);
		return fcb_idx;
	}
	else{
		// find empty space for writting a new file
		if(op == G_WRITE){
			// search for the empty fcb
			fcb_idx = find_empty_fcb(fs);
			// printf("empty fcb = %d\n", fcb_idx);

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
			*target_create_time = gtime/256;
			*(target_create_time + 1) = gtime%256;
			*target_modify_time = gtime/256;
			*(target_modify_time + 1) = gtime%256;
			*target_bit_size = 0;
			*(target_bit_size+1) = 1;
			*target_file_size = 0;
			*(target_file_size+1) = 0;
			*target_fcb_idx = fcb_idx;
			// find empty bit
			int empty_bit = find_empty_bit(fs);
			// printf("empty_bit = %d\n", empty_bit);
			*target_bit_location = empty_bit/256;
			*(target_bit_location + 1) = empty_bit%256;

			// occupy the empty bit
			uchar *bit = fs->volume + empty_bit/8;
			*bit += 1<<(empty_bit%8);

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
	uchar *target_bit_location = target_fcb + 28;

	int bit_location = (*target_bit_location) * 256 + *(target_bit_location+1);

	uchar *reader = fs->volume + 
					fs->SUPERBLOCK_SIZE + 
					(fs->FCB_ENTRIES*fs->FCB_SIZE) + 
					fs->STORAGE_BLOCK_SIZE*(bit_location);
	// printf("[read]\n");
	for(int i = 0; i < size; i++){
		*output = *reader;
		// printf("%d ", *output);
		reader ++;
		output++;
	}
	// printf("\n");
}

__device__ u32 find_fcb_from_bit(FileSystem *fs, int bit_num){
	// given the bit_num, return the corresponding fcb
	uchar *start_fcb = fs->volume + fs->SUPERBLOCK_SIZE;
	for(int i = 0; i<fs->FCB_ENTRIES; i++){
		uchar *target_bit_location = start_fcb + i*fs->FCB_SIZE + 28;
		int bit_locaiton = *(target_bit_location)*256 + *(target_bit_location+1);
		if(bit_locaiton == bit_num){
			printf("find the fcb of the bit\n");
			return i;
		}
	}
	printf("ERROR: NO FCB OF THE BIT\n");
}

__device__ void compact_storage(FileSystem *fs){
	printf("\nCompact this Shit\n");
	
	uchar *bitBlock = fs->volume;
	int cnt = 0;
	int emtpy_bit = 0;

	// find all empty bit shit
	while(cnt < fs->SUPERBLOCK_SIZE*8){
		int bit = ((*bitBlock)>>(cnt%8))%2;
		if(bit == 0){

			// find non-empty start
			int head_idx = cnt;
			uchar *findBlock;
			if(head_idx%8 == 7){
				findBlock = bitBlock + 1;
			}
			else{
				findBlock = bitBlock;
			}
			int head_value = 0;
			while(head_value == 0){
				// printf("find ass hole in %d\n", head_idx);
				head_idx++;

				if(head_idx >=fs->SUPERBLOCK_SIZE*8){
					printf("This stupid shit go to hell!!!\n");
					return;
				}

				if(head_idx%8 == 0){
					findBlock ++;
				}
				head_value = ((*findBlock>>(head_idx%8)))%2;
			}

			int head_fcb_num = find_fcb_from_bit(fs, head_idx);
			uchar *head_fcb = fs->volume + fs->SUPERBLOCK_SIZE + head_fcb_num*fs->FCB_SIZE;

			printf("head fcb = %s\n", head_fcb);

			uchar *target_bit_size = head_fcb + 26;
			uchar *target_bit_location = head_fcb + 28;

			*target_bit_location = cnt/256;
			*(target_bit_location + 1) = cnt%256;

			int bit_size = *(target_bit_size)*256 + *(target_bit_size+1);
			printf("bit_size = %d\n", bit_size);
			int end_idx = head_idx + bit_size;

			printf("this ass hole %d\n", cnt);
			printf("this shit fxxk at %d\n", head_idx);
			printf("this shit die at %d (not included)\n", end_idx);
			
			// change the bit map
			uchar *BLOCK = fs->volume + cnt/8;
			int offset = head_idx-cnt;

			for(int i = 0; i < offset; i++){
				*BLOCK += 1<<((cnt + i)%8);
				if((cnt + i)%8==7)
					BLOCK ++;
			}

			BLOCK = fs->volume + (end_idx -offset)/8;
			
			for(int j = 0; j < offset; j++){
				*BLOCK -= 1<<((end_idx -offset + j)%8);
				if((end_idx -offset + j)%8 == 7)
					BLOCK ++;
			}

			// transfer the storage
			for(int i = 0; i < bit_size; i++){
				uchar *storageBlock = fs->volume + 
										fs->SUPERBLOCK_SIZE+
										(fs->FCB_ENTRIES*fs->FCB_SIZE) +
										(head_idx + i)*fs->STORAGE_BLOCK_SIZE;

				uchar *emptyBlock = fs->volume +
										fs->SUPERBLOCK_SIZE+
										(fs->FCB_ENTRIES*fs->FCB_SIZE) +
										(cnt + i)*fs->STORAGE_BLOCK_SIZE;
				
				for(int j = 0; j < fs->STORAGE_BLOCK_SIZE; j++){
					if(i == 0)
						printf("cnt %d data: %d, head_idx %d data: %d",(cnt+i), *emptyBlock, head_idx+i, *storageBlock);
					*emptyBlock = *storageBlock;
					if(i == 0)
						printf("%d\n", *emptyBlock);
					emptyBlock ++;
					storageBlock ++;
				}
			}
			printf("Fuck me!!!\n");
			for(int i = 0; i < offset; i++){
				uchar *cleanBlock = fs->volume +
										fs->SUPERBLOCK_SIZE+
										(fs->FCB_ENTRIES *fs->FCB_SIZE) +
										(end_idx-offset+i)*fs->STORAGE_BLOCK_SIZE;

				for(int j = 0; j<fs->STORAGE_BLOCK_SIZE; j++){
					*cleanBlock = '\0';
					cleanBlock ++;
				}
			}
			printf("sTORAGE TRANSFERED!\n");

			cnt = end_idx-offset;
			bitBlock = fs->volume + cnt/8;
			bit = ((*bitBlock)>>(cnt%8))%2;
		}
		else{
			if(cnt %8 == 7)
				bitBlock ++;
			cnt ++;
		}
	}
}

__device__ u32 find_first_fit(FileSystem *fs, int bits_num){
	// if the file length is larger than the current length, find the first fit location.
	printf("[find_first_fit], bits number = %d\n", bits_num);

	// printf("FXXK try this bitch: %d\n", fs->try_bit_num);
	uchar *tryBlock = fs->volume + (fs->try_bit_num)/8;
	bool try_success = true;
	for(int i = 0; i < bits_num; i++){
		// printf("I want to fxxk u %d\n", i);
		int bit = (*tryBlock>>(fs->try_bit_num + i)%8)%2;
		if(bit == 1){
			try_success = false;
			break;
		}
		if((fs->try_bit_num + i)%8 == 7){
			tryBlock ++;
		}
	}
	if(try_success){
		// printf("BETTER THAN U IDIOT = %d\n",fs->try_bit_num );
		return fs->try_bit_num;
	}

	uchar *start_bit = fs->volume;
	int step = 0;
	while(step < fs->SUPERBLOCK_SIZE*8){

		uchar *block = start_bit + (step/8);
		int i;
		for(i = 0; i < bits_num; i++){
			// printf("BLOCK VALU = %d\n", *block);
			int bit = (*block >> (step%8))%2;
			if(bit == 1){
				// printf("step:%d in block:%d -> bit:%d OCCUPIED\n", step, step/8, step%8);
				step ++;
				break;
			}
			else{
				// printf("BLOCK = %d\n", *block);
				// printf("step:%d in block:%d -> bit:%d OK\n", step, step/8, step%8);
				if(step%8 == 7){
					block ++;
				}
				step ++;
			}
		}

		if(i == bits_num){
			// step++;
			printf("find! first fit bit = %d\n", step-bits_num);
			return step-bits_num;
		}
	}
	// no feasible storage space
	printf("NO FEASIBLE STORAGE, DO COMPACT\n");
	compact_storage(fs);
	return find_empty_bit(fs);
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

	// update file size
	*target_file_size = size/256;
	*(target_file_size+1) = size%256;

	// find suitable physical location
	int need_bit_size = ceil((float)(size)/(float)(fs->STORAGE_BLOCK_SIZE));
	int bit_size = *(target_bit_size)*256 + *(target_bit_size + 1);
	if(bit_size<need_bit_size){
		// if we need more space to store the data
		// set current bit map to 0, free the space in the bit-map.
		int bit_location = (*target_bit_location) * 256 + *(target_bit_location+1);

		uchar *free_block = fs->volume + bit_location/8;
		for(uchar i = 0; i < bit_size; i++){
			*free_block -= (1<<((bit_location+i)%8));
			if(((bit_location+i)%8) == 7){
				free_block ++;
			}
		} 
		// record the fs->try_bit_num
		fs->try_bit_num = bit_location;

		// reset the data in the STORAGE
		uchar *start_block = fs->volume + 
							fs->SUPERBLOCK_SIZE+
							(fs->FCB_ENTRIES*fs->FCB_SIZE) + 
							bit_location*(fs->STORAGE_BLOCK_SIZE);

		for(uchar b = 0; b < bit_size; b++){
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

		// printf("first_fit = %d\n", (*target_bit_location)*256+*(target_bit_location+1));

		// set bit_length to need_bit_sizef[]
		*target_bit_size = need_bit_size/256;
		*(target_bit_size+1) = need_bit_size%256;
		// occupy bit map
		uchar *bit_block = fs->volume + (first_fit/8);
		for(int i = 0; i < need_bit_size; i++){
			// printf("original value = %d", *bit_block);
			*bit_block += 1<<((first_fit+i)%8);
			// printf("[occupy:%d] block:%d->bit:%d, value = %d\n",first_fit+i, (first_fit+i)/8, (first_fit+i)%8, *bit_block);
			if((first_fit+i)%8 == 7){
				// printf("SHITSHITSHITSHITSHITSHITSHITSHITSHITSHITSHITSHITSHITSHITSHITSHITSHITSHITSHIT\n");
				bit_block ++;
			}
		}	
	}
	else if(bit_size< need_bit_size){
		// if the need_bit_size < current bit size, we set some bit to 0 to free some space in bit map
		uchar *free_block = fs->volume + (*target_bit_location)/8;
		for(int i = *target_bit_location+need_bit_size; i<*target_bit_location+bit_size; i++){
			*free_block -= (1<<(i%8));
			if(i%8 == 7){
				free_block ++;
			}
		} 
		*target_bit_size = need_bit_size/256;
		*(target_bit_size+1) = need_bit_size%256;
	}

	int bit_location = *(target_bit_location)*256+*(target_bit_location+1);

	uchar *writer = fs->volume + 
					fs->SUPERBLOCK_SIZE+
					fs->FCB_ENTRIES*fs->FCB_SIZE+
					fs->STORAGE_BLOCK_SIZE*  bit_location;

	// write data into STORAGE

	printf("[write data]:size = %d at bit location: %d\n", *(target_file_size)*256 + *(target_file_size+1), bit_location);
	for(int i = 0; i < size; i++){

		*writer = *input;

		writer ++;
		input++;
	}
	printf("\n");

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
			uchar *max_f = fcb_arr[i];
			int max_size = *(max_f+24)*256 + *(max_f+25);
			// printf("[CHECK SIZE] = %d\n", max_size);
			int max_idx = i;
			for(int j = i+1; j < fcb_num; j++){
				uchar *f = fcb_arr[j];
				int f_size = *(f+24)*256+*(f+25);
				if(f_size>max_size){
					max_size = f_size;
					max_idx = j;
				}
			}
			uchar *fcb1 = fcb_arr[i];
			uchar *fcb2 = fcb_arr[max_idx];
			fcb_arr[i] = fcb2;
			fcb_arr[max_idx] = fcb1; 
		}

		// sort by create time
		uchar *tmp = fcb_arr[0];
		uchar *next = fcb_arr[1];
		int cnt = 0;
		while(cnt < fcb_num-1){
			int tmp_size = *(tmp + 24)*256 + *(tmp+25);
			int next_size = *(next + 24)*256 + *(next + 25);
			if(tmp_size == next_size){
				printf("%s size = %d, %s size = %d\n",tmp, tmp_size, next, next_size);
				// find the end
				int same_size = tmp_size;
				int f_num = 0;				// the number of files in the sub-array
				uchar *sub_arr[1024];
				sub_arr[0] = tmp;
				f_num ++;
				while((*(next + 24)*256 + *(next + 25)) == same_size){
					sub_arr[f_num] = next;
					next += fs->FCB_SIZE;
					f_num ++;
				}

				// sort
				for(int i = 0; i < f_num-1; i++){
					uchar * fi = sub_arr[i];
					int min_time = *(fi+20)*256 + *(fi+21);
					int min_idx = i;
					for(int j = i+1; j < f_num; j++){
						uchar *fj = sub_arr[j];
						int time_j = *(fj+20)*256 + *(fj+21);
						if(time_j < min_time){
							min_idx = j;
							min_time = time_j;
						}
					}
					uchar *fcb1 = sub_arr[i];
					uchar *fcb2 = sub_arr[min_idx];

					fcb_arr[cnt+i] = fcb2;
					fcb_arr[cnt+min_idx] = fcb1; 

					sub_arr[cnt+i] = fcb2;
					sub_arr[cnt+min_idx] = fcb1; 
				}
				printf("[SUB ARRAY]len = %d\n", f_num);
				cnt += f_num;
				tmp = fcb_arr[cnt];
				next = fcb_arr[cnt+1];
			}
			else{
				tmp +=fs->FCB_SIZE;
				next +=fs->FCB_SIZE;
				cnt ++;
			}
		}

		for(int i = 0; i<fcb_num; i++){
      		uchar *fcb_cur = fcb_arr[i];
			int len = *(fcb_cur+24)*256+*(fcb_cur+25);
			int creat_time = *(fcb_cur+20)*256+*(fcb_cur+21);
      		printf("%s len = %d, create time = %d\n",fcb_cur, len, creat_time);		
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
			file_exist = true;
			fcb_idx = i;
		}
	}

	if(file_exist == false){
		printf("ERROR: FILE DOES NOT EXIST!\n");
		return;
	}

	uchar *target_fcb = start_fcb + fcb_idx*fs->FCB_SIZE;
	uchar *target_bit_size = target_fcb + 26;
	uchar *target_bit_location = target_fcb + 28;

	int bit_location = (*target_bit_location)*256 + *(target_bit_location+1);
	// clean the bit map
	uchar *rm_block = fs->volume + bit_location/8;
	printf("original value = %d ", *rm_block);

	int bit_size = *(target_bit_size)*256 + *(target_bit_size+1);

	for(int i = 0; i < bit_size; i++){
		*rm_block -= 1<<((bit_location+i)%8);
		printf("[RM bit] block:%d->bit:%d value = %d\n", (bit_location+i)/8, (bit_location+i)%8, *rm_block);
		
		if(((*target_bit_location)%8)+i == 7){
			rm_block ++;
		}
	}

	// record the fs->try_bit_num
	fs->try_bit_num = bit_location;

	// clean the STORAGE
	uchar *rm_storage = fs->volume + fs->SUPERBLOCK_SIZE + 
						(fs->FCB_ENTRIES*fs->FCB_SIZE);

	for(int b = 0; b < bit_size; b++){
		for(int i = 0; i < fs->STORAGE_BLOCK_SIZE; i++){
			*rm_storage = 0;
			rm_storage ++;
		}
	}

	// set RM fcb to '\0'
	for(int i = 0; i < fs->FCB_SIZE; i++){
		if(*target_fcb == '\0')
			break;
		*target_fcb = 0; 
		target_fcb ++;
	}
}

