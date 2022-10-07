#include <stdio.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/wait.h>
#include <sys/types.h>

int main(int argc, char *argv[])
{
    pid_t pid;
    int status;
    
    printf("process start to fork\n");
    pid = fork();

    // child
    if(pid == 0){
        printf("child...\n");
        printf("raising SIGCHILD signal...\n\n");
        raise(SIGCHLD);     //Raise SIGCHLD in child process
    }

    // parent
    else{
        wait(&status);
        printf("Parent process receives the signal\n");

        if(WIFEXITED(status)){  //Check if child process exits normally
            printf("Normal termination with EXIT STATUS = %d\n", WEXITSTATUS(status));   //Get status value of child process
        }else if(WIFSIGNALED(status)){
            printf("CHILD EXECUTION FAILED:%d\n", WTERMSIG(status));
        }else if(WIFSTOPPED(status)){
            printf("CHILD PROCESS STOPPED: %d\n", WSTOPSIG(status));
        }else{
            printf("CHILD PROCESS CONTINUED\n");
        }
        exit(0);
    }
    return 0;
}
