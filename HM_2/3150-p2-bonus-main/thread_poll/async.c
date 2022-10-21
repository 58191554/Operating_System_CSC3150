#include <stdlib.h>
#include <pthread.h>
#include "async.h"
#include "utlist.h"


// thread_safe_queue 
struct my_queue work_queue;
int THREAD_NUM;

// create an array of threads
pthread_t * th;
pthread_mutex_t mutexQueue;

my_item_t taskQueue[5000];
int taskCount = 0;

// check if any thread in the thread_arr is free.
int try_pop(struct my_queue work_queue){
    // return 1 if true.
    return 0;
}
void executeTask(my_item_t *task){
    task->taskFunction(task->args);
}


void submitTask(my_item_t task){
    pthread_mutex_lock(&mutexQueue);
    taskQueue[taskCount] = task;
    taskCount ++;
    pthread_mutex_unlock(&mutexQueue);
}


void* startThread(void* args){
    while(1){
        my_item_t *task;

        pthread_mutex_lock(&mutexQueue);
        if(work_queue.size > 0){
            task = work_queue.head;
            int i ;
            // TODO try pop the task from the my_queue
            work_queue.size--;
        }
        pthread_mutex_unlock(&mutexQueue);

        executeTask(task);
    }
}


void async_init(int num_threads) {
    /** TODO: create num_threads threads and initialize the thread pool **/

    pthread_mutex_init(&mutexQueue, NULL);


    THREAD_NUM = num_threads;
    pthread_t th[num_threads];
    // need a array to store the thread in the thread pool
    for (int i = 0; i < num_threads; i++){
        if(pthread_create(&th[i], NULL, &startThread, NULL) != 0){
            // perror("Failed to create the thread");
        }
    }
    // a joiner object to join

    return ;
}   

void async_run(void (*hanlder)(int), int args) {

    // int i;
    // my_item_t t = {
    //     .taskFunction = hanlder,
    //     .args = args
    // };

    // // submitTask the task my_item_t t
    // submitTask(t);

    return ;
}
