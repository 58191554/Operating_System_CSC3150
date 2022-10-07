#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <string.h>

int main(int argc, char const *argv[])
{
    char buf[50] = "Original test strings";
    pid_t pid;
    printf("Process start to fork\n");
    pid = fork();

    if (pid == -1){
        perror("fork");
        exit(1);
    }
    else{
        // Child process
        if(pid == 1){
            strcpy(buf, "Test strings are updated by child.");
            printf("I'm the child Process %s\n", buf);
            exit(0);
        }
        // Parent process
        else{
            sleep(3);
            printf("I'm the Parent Process:\n");
            printf("\t My pid is:%d\n", getpid());
            exit(0);
        }
    }
    return 0;
}
