#include <stdlib.h>
#include <pthread.h>
#include "async.h"
#include "utlist.h"

// an array of threads
pthread_t * th;
pthread_mutex_t mutexQueue;
// a queue of tasks
my_queue_t workQueue ;
my_queue_t *workQueuePt = &workQueue;

void submitTask(my_item_t *task){
    pthread_mutex_lock(&mutexQueue);
    enqueue(workQueuePt, task);
    printf("the queue size is %d\n", workQueue.size);
    pthread_mutex_unlock(&mutexQueue);
}

void enqueue(my_queue_t *queue, my_item_t *item){
    if(queue->size == 0){
        queue->head = item;
    }
    else{
        queue->tail->next = item;
        queue->tail = item;
    }
    queue->size ++;
}

my_item_t * dequeue(my_queue_t *queue){
    if(queue->size == 0){
        return NULL;
    }
    else{
        int last = queue->tail;
        queue->tail = queue->tail->prev;
        return last;
    }
    queue->size --;
}

void executeTask(my_item_t * task){
    printf("task = %d", task->taskFunction);
    printf("args = %d", task->args);
    task->taskFunction(task->args);
}

void* startThread(void* args){
    while(1){
        my_item_t *task;

        pthread_mutex_lock(&mutexQueue);
        if(workQueuePt->size > 0){
            printf("myQueue size = %d", workQueuePt->size);
            task = dequeue(workQueuePt->head);
            workQueuePt->size--;
        }
        pthread_mutex_unlock(&mutexQueue);
        executeTask(task);
    }
}

void async_init(int num_threads) {
    /** TODO: create num_threads threads and initialize the thread pool **/
    workQueue.head = NULL;
    workQueue.size = 0;
    pthread_mutex_init(&mutexQueue, NULL);
    pthread_t th[num_threads];
    // need a array to store the thread in the thread pool
    int i;

    for (i = 0; i < num_threads; i++){
        if(pthread_create(&th[i], NULL, &startThread, NULL) != 0){
            perror("Failed to create the thread");
        }
    }

    return ;
}   

void async_run(void (*hanlder)(int), int args) {
    // submitTask the task my_item_t t
    my_item_t t = {
        .taskFunction = hanlder,
        .args = args
    };

    submitTask(&t);

    return ;
}
