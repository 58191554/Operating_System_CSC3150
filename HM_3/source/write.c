#include <stdio.h>

#define OUTPUTFILE "./order.bin"
#define SIZE (1 << 17)


void write_binaryFile(char *fileName, void *buffer, int bufferSize) {
  	FILE *fp;
  	fp = fopen(fileName, "wb");
  	fwrite(buffer, 1, bufferSize, fp);
  	fclose(fp);
}

int main(int argc, char const *argv[])
{

    int input[SIZE];
    for(int i = 0;i<SIZE;i++)
        input[i] = i;
    write_binaryFile(OUTPUTFILE,input, SIZE);
    return 0;
}
