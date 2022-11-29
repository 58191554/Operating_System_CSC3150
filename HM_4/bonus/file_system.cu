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

	// init the root directory
	fs->cur_DIRfcb_idx = 0;
	uchar *dir_fcb = fs->volume + fs->SUPERBLOCK_SIZE;
	uchar *name = dir_fcb;
	*name = 'r'; name++;
	*name = 'o'; name++;
	*name = 'o'; name++;
	*name = 't'; name++;
	*name = '\0';

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
	// printf("current path  = %s\n", fs->volume + fs->SUPERBLOCK_SIZE + fs->cur_DIRfcb_idx * fs->FCB_SIZE);
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
	// directory
	uchar *dir_fcb = fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fs->cur_DIRfcb_idx;
	int file_num = *(dir_fcb + 26);
	int son_idx = *(dir_fcb+24)*256 + *(dir_fcb+25);

	if(file_exist){
		// Although some b.txt exists, need to check whether these files are in this path. 
		// Use for to traverse each time to check fcb_idx if the same fcb_idx is found
		uchar * bro_fcb = fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * son_idx;
		for(int i = 0; i < file_num; i++){
			if(son_idx == fcb_idx){
				return fcb_idx;
			}
			son_idx = *(bro_fcb+30)*256 + *(bro_fcb + 31);
			bro_fcb = fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * son_idx;
		}
	}
	
	// find empty space for writting a new file
	if(op == G_WRITE){
		
		// son number +1 ;
		*(dir_fcb+26) += 1;

		fcb_idx = find_empty_fcb(fs);
		if(file_num == 0){
			// add first son
			*(dir_fcb+24) = fcb_idx/256;
			*(dir_fcb+25) = fcb_idx%256;
		}
		else{
			son_idx = *(dir_fcb+24)*256 + *(dir_fcb+25);
			uchar *son_fcb = fs->volume + fs->SUPERBLOCK_SIZE + son_idx*fs->FCB_SIZE;
			for(int i = 0; i < file_num-1; i++){
				son_idx = *(son_fcb+30)*256 + *(son_fcb+31);
				son_fcb = fs->volume + fs->SUPERBLOCK_SIZE + son_idx*fs->FCB_SIZE;
			}
			*(son_fcb+30) = fcb_idx/256;
			*(son_fcb+31) = fcb_idx%256;
		}

		uchar *target_fcb=start_fcb + fcb_idx * fs->FCB_SIZE;
		uchar *target_name = target_fcb;
		// write file name
		for(char *c = s; *c!='\0'; c++){
			*target_name = *c;
			target_name ++;
		}
		*target_name = '\0'; 
		gtime++;
		uchar *target_create_time = target_fcb + 20;
		uchar *target_modify_time = target_fcb + 22;
		*target_create_time = gtime/256;
		*(target_create_time + 1) = gtime%256;
		*target_modify_time = gtime/256;
		*(target_modify_time + 1) = gtime%256;

		// check it is a file or a dir
		bool dir_flag = true;
		for(char *c = s; *c!='\0'; c++){
			if(*c == '.'){
				// is a file not a directory
				dir_flag = false;
				break;
			}
			s++;
		}
		if(dir_flag){
			// is a dir
			uchar *son_fcb_idx = target_fcb + 24;
			uchar *son_number = target_fcb + 26;
			uchar *father = target_fcb + 28;
			*father = fs->cur_DIRfcb_idx/256;
			*(father +1) = fs->cur_DIRfcb_idx%256;
		}
		else{
			// is a file
			uchar *target_file_size = target_fcb + 24;
			uchar *target_bit_size = target_fcb + 26;
			uchar *target_bit_location = target_fcb + 28;

			*target_bit_size = 0;
			*(target_bit_size+1) = 1;
			*target_file_size = 0;
			*(target_file_size+1) = 0;
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
	}
	else if(op == G_READ){
		printf("No readable file named %s\n", s);
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
			// printf("find the fcb of the bit\n");
			return i;
		}
	}
	printf("ERROR: NO FCB OF THE BIT\n");
}

__device__ void compact_storage(FileSystem *fs){
	
	uchar *bitBlock = fs->volume;
	int cnt = 0;
	int emtpy_bit = 0;

	// find all empty bit 
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
				head_idx++;

				if(head_idx >=fs->SUPERBLOCK_SIZE*8){
					return;
				}

				if(head_idx%8 == 0){
					findBlock ++;
				}
				head_value = ((*findBlock>>(head_idx%8)))%2;
			}

			int head_fcb_num = find_fcb_from_bit(fs, head_idx);
			uchar *head_fcb = fs->volume + fs->SUPERBLOCK_SIZE + head_fcb_num*fs->FCB_SIZE;

			// printf("head fcb = %s\n", head_fcb);

			uchar *target_bit_size = head_fcb + 26;
			uchar *target_bit_location = head_fcb + 28;

			*target_bit_location = cnt/256;
			*(target_bit_location + 1) = cnt%256;

			int bit_size = *(target_bit_size)*256 + *(target_bit_size+1);
			int end_idx = head_idx + bit_size;
	
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
					*emptyBlock = *storageBlock;
					emptyBlock ++;
					storageBlock ++;
				}
			}
			// printf("Fuck me!!!\n");
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
			// printf("sTORAGE TRANSFERED!\n");

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
	// printf("[find_first_fit], bits number = %d\n", bits_num);

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
			// printf("find! first fit bit = %d\n", step-bits_num);
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

	// find the target fcb by linkedlist


	uchar *target_fcb = fs->volume + fs->SUPERBLOCK_SIZE + fp*fs->FCB_SIZE;
	uchar *target_modify_time = target_fcb + 22;
	uchar *target_file_size = target_fcb + 24;
	uchar *target_bit_size = target_fcb + 26;
	uchar *target_bit_location = target_fcb + 28;

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

		// transfer int to uchar
		*target_bit_location = first_fit/256;
		*(target_bit_location+1) = first_fit%256;

		// set bit_length to need_bit_sizef[]
		*target_bit_size = need_bit_size/256;
		*(target_bit_size+1) = need_bit_size%256;
		// occupy bit map
		uchar *bit_block = fs->volume + (first_fit/8);
		for(int i = 0; i < need_bit_size; i++){
			*bit_block += 1<<((first_fit+i)%8);
			if((first_fit+i)%8 == 7){
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

	// printf("[write data]:size = %d at bit location: %d\n", *(target_file_size)*256 + *(target_file_size+1), bit_location);
	for(int i = 0; i < size; i++){

		*writer = *input;

		writer ++;
		input++;
	}
	// printf("\n");

	// modify time update
	gtime ++;
	*(target_modify_time) = gtime/256;
	*(target_modify_time +1) = gtime%256;

	return fp;
}

__device__ bool is_dir(FileSystem *fs, uchar *fcb){
	// input a fcb, check its name to know if it is directory
	uchar *name_char = fcb;
	for(int i = 0; i < 20; i++){
		if(*name_char == '.')
			return false;
		name_char ++;
	}
	return true;
}

__device__ int get_dir_size(FileSystem *fs, uchar *dir_fcb){
	// output the size of the dir shit
	int dir_size = 0;
	int son_num = *(dir_fcb+26);
	int cnt = 0;
	int son_fcb_idx = *(dir_fcb+24)*256 + *(dir_fcb+25);

	while(cnt < son_num){
		uchar *son_fcb = fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * son_fcb_idx;

		int file_size = 0;
		for(uchar *c = son_fcb; *c!='\0'; c++){
			// printf("%c\n", *c);
			file_size ++;
		}
		file_size ++;
		dir_size += file_size;
		cnt ++;
		son_fcb_idx = *(son_fcb+30)*256 + *(son_fcb + 31);
	}
	return dir_size;
}

__device__ void fs_gsys(FileSystem *fs, int op)
{
	/* Implement LS_D and LS_S operation here */
	// get the fcb number and fcb array

	uchar *dir_fcb = fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fs->cur_DIRfcb_idx;
	int son_num = *(dir_fcb + 26);
	// printf("son_num = %d\n", son_num);
// 
	uchar *fcb_arr[1024];
	int son_idx = *(dir_fcb+24)*256 + *(dir_fcb + 25);

	uchar *son_fcb = fs->volume + fs->SUPERBLOCK_SIZE + son_idx*fs->FCB_SIZE;

	int cnt = 0;
	while(cnt < son_num){
		// printf("ADD FILE = %s", son_fcb);
		fcb_arr[cnt] = son_fcb;
		cnt ++;
		son_idx = *(son_fcb+30)*256 + *(son_fcb + 31);
		son_fcb = fs->volume + fs->SUPERBLOCK_SIZE + + son_idx*fs->FCB_SIZE;
	}

	if(op == LS_D){
		printf("\n===sort by modified time===\n");
		// // use bubble sort
		for(int i = 0; i < son_num-1; i++){
			int m_time_latest = *(fcb_arr[i] + 22)*256 + *(fcb_arr[i]+23);
			int max_idx = i;
			for(int j = i+1; j < son_num; j++){
				int m_time_j = *(fcb_arr[j] + 22)*256 + *(fcb_arr[j]+23);
				if(m_time_j>m_time_latest){
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
		for(int i = 0; i<son_num; i++){
      		uchar *fcb_cur = fcb_arr[i];
      		printf("%s ",fcb_cur);		 
			if(is_dir(fs, fcb_cur))
				printf("d\n");
			else
				printf("\n");
		}
	}

	if(op == LS_S){
		printf("\n===sort by file size===\n");

		for(int i = 0; i < son_num-1; i++){
			uchar *max_f = fcb_arr[i];
			int max_size;
			// if shit is dir
			if(is_dir(fs, max_f)){
				max_size = get_dir_size(fs, max_f);
			}
			else{
				max_size = *(max_f+24)*256 + *(max_f+25);
			}
			int max_idx = i;
			for(int j = i+1; j < son_num; j++){
				uchar *f = fcb_arr[j];
				int f_size;
				if(is_dir(fs, f)){
					f_size = get_dir_size(fs, f);
				}
				else{
					f_size = *(f+24)*256+*(f+25);
				}
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

		// // sort by create time
		uchar *tmp = fcb_arr[0];
		uchar *next = fcb_arr[1];
		int cnt = 0;
		while(cnt < son_num-1){

			// printf("CNT = %d\n", cnt);
			uchar *tmp = fcb_arr[cnt];
			uchar *next = fcb_arr[cnt+1];

			int tmp_size = *(tmp + 24)*256 + *(tmp+25);
			int next_size = *(next + 24)*256 + *(next + 25);

			if(tmp_size == next_size){

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

					sub_arr[i] = fcb2;
					sub_arr[min_idx] = fcb1; 
				}
				// printf("[SUB ARRAY]len = %d\n", f_num);
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

		for(int i = 0; i<son_num; i++){
      		uchar *fcb_cur = fcb_arr[i];
			int len;
			int creat_time = *(fcb_cur+20)*256+*(fcb_cur+21);
      		// printf("%s file size = %d, create time = %d\n",fcb_cur, len, creat_time);		
			if(is_dir(fs, fcb_cur)){
				len = get_dir_size(fs, fcb_cur);
      			printf("%s %d d\n",fcb_cur, len);

			}
			else{
				len = *(fcb_cur+24)*256 + *(fcb_cur+25);
      			printf("%s %d \n",fcb_cur, len);
			}
		}
	}

	if(op == PWD){
		uchar *father_arr[1024];
		int father_num = 0;
		uchar *now_dir_fcb = fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fs->cur_DIRfcb_idx;
		int father_fcb_idx = *(now_dir_fcb+28)*256 + *(now_dir_fcb+29);

		// if this shit is in root dir
		if(father_fcb_idx == 0){
			printf("/%s\n", now_dir_fcb);
			return;
		}

		uchar *father_fcb = fs->volume + fs->SUPERBLOCK_SIZE + father_fcb_idx * fs->FCB_SIZE;
		// add first father in father array
		father_arr[father_num] = father_fcb;
		father_num ++;
		// printf("ADD FATHER %s\n", father_fcb);

		father_fcb_idx = *(father_fcb+28)*256 + *(father_fcb+29);
		// find father until we find root
		while(father_fcb_idx != 0){
			father_arr[father_num] = father_fcb;
			father_num ++;
			// printf("ADD FATHER %s\n", father_fcb);
			father_fcb  = fs->volume + fs->SUPERBLOCK_SIZE + father_fcb_idx * fs->FCB_SIZE;
			father_fcb_idx = *(father_fcb+28)*256 + *(father_fcb+29);
		}
		// output
		for(int i = father_num-1; i >= 0; i--){
			printf("/%s", father_arr[i]);
		}
		printf("/%s\n", now_dir_fcb);
	}

	if(op == CD_P){
		uchar *cur_dir_fcb = fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fs->cur_DIRfcb_idx;
		int father_idx = *(cur_dir_fcb+28)*256 + *(cur_dir_fcb+29);
		fs->cur_DIRfcb_idx = father_idx;
	}

}

__device__ void remove_sons(FileSystem *fs, uchar *fcb){
	// delete the sons in the dir
	uchar *del_arr[1024];
	int son_num = *(fcb+26);
	int son_idx = *(fcb+24)*256 + *(fcb+25);
	uchar *son_fcb = fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * son_idx;

	// add current files into del array
	for(int i = 0; i < son_num; i++){
		del_arr[i] = son_fcb;
		son_idx = *(son_fcb+30)*256 + *(son_fcb + 31);
		son_fcb = fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * son_idx;
	}

	// DELETE AND ADD NEW DIR TO DELETE ARRAY
	int del_num = son_num;
	int cnt = 0;
	while(cnt < del_num){
		uchar *del_fcb = del_arr[cnt];
		if(is_dir(fs, del_fcb)){
			// add its sons into the del_arr
			int del_son_num = *(del_fcb+26);
			int del_son_idx = *(del_fcb+24)*256 + *(del_fcb+25);
			uchar *del_son_fcb = fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * del_son_idx;

			// add current files into del array
			for(int i = 0; i < son_num; i++){
				del_num ++;
				del_arr[del_num] = del_son_fcb;
				del_son_idx = *(del_son_fcb+30)*256 + *(del_son_fcb + 31);
				del_son_fcb = fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * del_son_idx;
			}
		}
		else{
			// RM file
			uchar *target_bit_size = del_fcb + 26;
			uchar *target_bit_location = del_fcb + 28;
	
			int bit_location = (*target_bit_location)*256 + *(target_bit_location+1);
			// clean the bit map
			uchar *rm_block = fs->volume + bit_location/8;
			// printf("original value = %d ", *rm_block);
	
			int bit_size = *(target_bit_size)*256 + *(target_bit_size+1);
	
			for(int i = 0; i < bit_size; i++){
				*rm_block -= 1<<((bit_location+i)%8);
				// printf("[RM bit] block:%d->bit:%d value = %d\n", (bit_location+i)/8, (bit_location+i)%8, *rm_block);
	
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
		}
		// set zero in fcb
		for(int i = 0; i < fs->FCB_SIZE; i++){
			*(del_fcb) = 0;
			del_fcb++;
		}

		// cnt plus to loop
		cnt ++;
	}
}

__device__ void fs_gsys(FileSystem *fs, int op, char *s)
{
	/* Implement rm operation here */
	if(op == RM){

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
		// printf("original value = %d ", *rm_block);

		int bit_size = *(target_bit_size)*256 + *(target_bit_size+1);

		for(int i = 0; i < bit_size; i++){
			*rm_block -= 1<<((bit_location+i)%8);
			// printf("[RM bit] block:%d->bit:%d value = %d\n", (bit_location+i)/8, (bit_location+i)%8, *rm_block);

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

	if(op == MKDIR){
		int new_fcb_idx = find_empty_fcb(fs);
		uchar *new_fcb = fs->volume + fs->SUPERBLOCK_SIZE + new_fcb_idx * fs->FCB_SIZE;
		uchar *new_name = fs->volume + fs->SUPERBLOCK_SIZE + new_fcb_idx * fs->FCB_SIZE;

		// write the name of the directory 
		for(char *c = s; *c!='\0'; c++){
			*new_name = *c;
			new_name ++;
		}
		*new_name = '\0'; 

		uchar *create_time = new_fcb + 20;
		uchar *modify_time = new_fcb + 22;
		gtime++;
		*create_time = gtime/256;
		*(create_time + 1) = gtime%256;
		*modify_time = gtime/256;
		*(modify_time + 1) = gtime%256;

		uchar *father = new_fcb+28;

		// set father
		*father = fs->cur_DIRfcb_idx/256;
		*(father + 1) = fs->cur_DIRfcb_idx%256;

		uchar *dir_fcb = fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * fs->cur_DIRfcb_idx;

		// set bro
		int bro_idx = *(dir_fcb + 24)*256 + *(dir_fcb + 25);
		uchar *bro_fcb = fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * bro_idx;
		int bro_num = *(dir_fcb + 26);
		for(int i = 0; i < bro_num -1; i++){

			bro_idx = *(bro_fcb+30)*256 + *(bro_fcb+31);
			bro_fcb = fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * bro_idx;
		}
		*(bro_fcb+30) = new_fcb_idx/256;
		*(bro_fcb + 31) = new_fcb_idx%256;

		// father son number + 1
		*(dir_fcb + 26) += 1;
	}

	if(op == CD){
		// find the directory fcb
		bool dir_exist = false;
		uchar *start_fcb = fs->volume + fs->SUPERBLOCK_SIZE;
		// find the file name in FCB
		for(int i = 0; i < fs->FCB_ENTRIES; i++){
			uchar *fcb = start_fcb + i * fs->FCB_SIZE;
			if(check_name(fcb, s)){
				dir_exist = true;
				fs->cur_DIRfcb_idx = i;
			}
		}
	
		if(dir_exist == false){
			printf("ERROR: NO SUCH DIRECTORY\n");
			return;
		}
	}

	if(op == RM_RF){
		// find the file name in FCBs
		bool exist = false;
		int target_fcb_idx;
		for(int i = 0; i<fs->FCB_ENTRIES; i++){
			uchar *fcb = fs->volume + fs->SUPERBLOCK_SIZE + i*fs->FCB_SIZE;
			if(check_name(fcb, s)){
				exist = true;
				target_fcb_idx = i;
			}
		}

		if(exist == false){
			printf("ERROR: DIR DOES NOT EXIST!\n");
			return;
		}
		uchar *target_fcb = fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * target_fcb_idx;

		// remove the sons in the dir
		remove_sons(fs, target_fcb);

		// remove from its father's sons
		int father_idx = *(target_fcb+28)*256 + *(target_fcb+29);
		uchar *father = fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * father_idx;
		int son_num = *(father+26);
		int son_idx = *(father+24)*256 + *(father+25);
		uchar *son_fcb = fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * son_idx;
		int elder_bro_idx = 0;
		for(int i = 0; i < son_num; i++){
			if(son_idx == target_fcb_idx){
				// printf("ELDER IDX = %d\n", elder_bro_idx);
				// printf("find the son %s\n", fs->volume+ fs->SUPERBLOCK_SIZE + fs->FCB_SIZE*son_idx);
				// printf("target idx = %d\n", target_fcb_idx);
				break;
			}
			elder_bro_idx = son_idx;
			son_idx = *(son_fcb + 30)*256 + *(son_fcb + 31);
			son_fcb = fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * son_idx;
		}
		int little_bro_idx = *(son_fcb + 30)*256 + *(son_fcb + 31);
		uchar *elder_bro = fs->volume + fs->SUPERBLOCK_SIZE + fs->FCB_SIZE * elder_bro_idx;

		*(elder_bro+30) = little_bro_idx/256;
		*(elder_bro+31) = little_bro_idx%256;

		// father file number -1
		*(father+26) -= 1;

		// father modify time ++
		gtime ++;
		*(father+22)=gtime/256;
		*(father+23) = gtime%256;
	}
}

