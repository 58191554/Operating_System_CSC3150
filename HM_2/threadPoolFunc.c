#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>


#define THREAD_NUM 4

typedef struct Task
{
    // add function pointer
    void (*taskFunction)(int, int);
    int arg1, arg2;
} Task;

// task array as a queue
Task taskQueue[256];
int taskCount = 0;

pthread_mutex_t mutexQueue;
pthread_cond_t condQueue;

void sumAndProduct(){
    int a = rand()%100;
    int b = rand()%100;
    int sum = a + b;
    int prod = a*b;
    printf("sum = %d, product = %d\n", sum, prod);
}

// function to execute the task
void executeTask(Task *task){
    // int result = task->a + task->b;
    // printf("the sum is %d\n", result);
    task->taskFunction(task->arg1, task->arg2);
}

void submitTask(Task task){
    pthread_mutex_lock(&mutexQueue);
    taskQueue[taskCount] = task;
    taskCount ++;
    pthread_mutex_unlock(&mutexQueue);
    pthread_cond_signal(&condQueue);
}


void* startThread(void* args){
    while(1){
        Task task;

        pthread_mutex_lock(&mutexQueue);
        while(taskCount == 0){
            pthread_cond_wait(&condQueue, &mutexQueue);
        }
        if(taskCount > 0){
            task = taskQueue[0];
            int i ;
            for (i = 0; i < taskCount -1; i++){
                taskQueue[i] = taskQueue[i+1];
            }
            taskCount--;
        }
        pthread_mutex_unlock(&mutexQueue);

        executeTask(&task);
    }
}

int main(int argc, char* argv[]){

// initialize the thread array
    pthread_t th[THREAD_NUM];
    pthread_mutex_init(&mutexQueue, NULL);
    pthread_cond_init(&condQueue, NULL);
    int i;

    for (i = 0; i < THREAD_NUM; i++){
        if(pthread_create(&th[i], NULL, &startThread, NULL) != 0){
            perror("Failed to create the thread");
        }
    }

    for(i = 0; i < 100; i++){
        Task t = {
            .taskFunction = &sumAndProduct,
            .arg1 = rand()%100,
            .arg2 = rand()%100
        };
        submitTask(t);
    }

    for (i = 0; i < THREAD_NUM; i++){
        if(pthread_join(th[i],  NULL) != 0){
            perror("Failed to join the thread");
        }
    }
    pthread_mutex_destroy(&mutexQueue);
    pthread_cond_destroy(&condQueue);
    return 0;
}
