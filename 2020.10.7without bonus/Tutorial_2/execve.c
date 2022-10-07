#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <string.h>
#include <wait.h>

int main(int argc, char *argv[])
{
    int state;
    pid_t pid = fork();

    if(pid<0){
        printf("fork error!\n");
    }
    else{
        // child
        if(pid == 0){
            int i;
            char *arg[argc];
            printf("this is child process...\n");
            for ( i = 0; i < argc-1; i++){
                arg[i] = argv[i+1];
            }

            arg[argc-1] = NULL;
            printf("child process id is %d\n", getpid());
            printf("child process start to execute test program:\n");
            execve(arg[0], arg, NULL);

            printf("Continue to run original child process!\n");

            perror("execve");
            exit(EXIT_FAILURE);
        }

        // Parent process
        else{
            wait(&state);
            printf("This is father process.\n");
            exit(1);
        }
    }
    return 0;
}
