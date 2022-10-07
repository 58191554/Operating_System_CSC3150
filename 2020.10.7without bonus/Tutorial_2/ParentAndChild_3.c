#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <string.h>

int main(int argc, char const *argv[])
{
    pid_t pid;

    printf("Process start to fork\n");
    pid = fork();
    printf("pid = %d\n", pid);
    if(pid == -1){
        perror("fork");
        exit(1);
    }

    else{
        // child process
        if(pid == 0){
            printf("I'm the Child Process, my pid = %d, my ppid = %d\n", getpid() , getppid());
            exit(0);
        }
        // parent process
        else{
            // sleep(3);   Parent and cild process runs concurrently after forking.
            printf("I'm the Parent Process, pid = %d\n", getpid());
            exit(0);
        }
    }

    return 0;    
}
