#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <wait.h>

int main(int argc, char *argv[]) {

  int status;

  printf("\nProcess start to fork\n");
  pid_t pid = fork();

  if (pid < 0) {
    printf("fork error!\n");
  } else {
    // child
    if (pid == 0) {
      char *arg[argc];
      for (int i = 0; i < argc - 1; i++) {
        arg[i] = argv[i + 1];
      }
      arg[argc - 1] = NULL;

      printf("I'm the Child Process, my pid = %d\n", getpid());
      printf("Child process start to execute test program:\n");
      execve(arg[0], arg, NULL);
      perror("execve");
      exit(SIGCHLD);
    }

    // Parent process
    else {
      printf("I'm the Parent Process, my pid = %d\n", getpid());

      waitpid(pid, &status, WUNTRACED);
      printf("Parent process receives the SIGCHLD signal\n");
      if (WIFEXITED(status)) { // if the child return normal
        printf("Normal termination with EXIT STATUS = %d\n",
               WEXITSTATUS(status));
      } else if (WIFSTOPPED(status)) {
        printf("CHILD PROCESS STOPPED:%d\n", WSTOPSIG(status));
      } else if (WIFSIGNALED(status)) {
        printf("CHILD PROCESS FAILED:%d\n", WTERMSIG(status));
      } else {
        printf("CHILD PROCESS CONTINUED\n");
      }
      exit(0);
    }
  }
  return 0;
}
