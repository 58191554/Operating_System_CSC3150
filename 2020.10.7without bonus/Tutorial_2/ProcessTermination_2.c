#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{

    pid_t pid;
    int status;

    printf("Process start to fork\n");
    pid = fork();

    if(pid == -1){
        perror("fork");
        exit(1);
    }
    else{
        // child
        if(pid == 0){
            printf("I'm the child process\n");
            sleep(10);
            printf("my pid = %d, my ppid = %d\n", getpid(), getppid());
            exit(0);
        }
        // parent
        else{
            waitpid(pid, &status, 0);
            printf("I'm the parent process\n");
            printf("my pid = %d\n", getpid());
            printf("child process exited with status %d\n", status);
            exit(0);
        }
    }
    return 0;
}
