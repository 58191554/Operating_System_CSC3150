#include <stdlib.h>
#include <pthread.h>
#include "async.h"
#include "utlist.h"

// an array of threads

void enqueue(my_queue_t *queue, my_item_t *item){
    printf("enqueue, the queue size is %d\n", queue->size);
    if(queue->size == 0){
        queue->head = item;
        queue->tail = item;
    }
    else{
        queue->tail->next = item;
        queue->tail = item;
    }
    queue->size ++;
}

my_item_t * dequeue(my_queue_t *queue){
    printf("dequeue, the queue size is %d\n", queue->size);

    if(queue->size == 0){
        return NULL;
    }
    else{
        int last = queue->tail;
        // if only one node in queue
        if(queue->size == 1){
            queue->head = NULL;
            queue->tail = NULL;
        }
        // if more than 1 node in queue
        else{
            queue->tail = queue->tail->prev;
        }
        queue->size --;
        return last;
    }
}

void* startThread(void* args){
    printf("startThread\n");
    while(1){
        my_item_t *task;
        if(work_q_pt->size >0){
            // pop the task from the queue
            pthread_mutex_lock(&mutex);
            while(work_q_pt->size == 0){
                pthread_cond_wait(&cond, &mutex);
            }
            if(work_q_pt->size > 0){
                printf("queue size = %d\n", work_q_pt->size);
                task = work_q_pt->head;
                DL_DELETE(work_q_pt->head, work_q_pt->head);
                work_q_pt->size--;
                printf("dequeue, queue size = %d\n", work_q_pt->size);
            }
            pthread_mutex_unlock(&mutex);

            // do the task
            printf("task begin...\n");
            task->taskFunction(task->args);
            printf("task done...\n");
        }
    }
}

void async_init(int num_threads) {
    /** TODO: create num_threads threads and initialize the thread pool **/
    my_queue_t work_queue = {
        .size = 0,
        .head = NULL,
        .tail = NULL
    };
    work_q_pt = (my_queue_t*)malloc(sizeof(work_queue));

    pthread_mutex_init(&mutex, NULL);
    pthread_cond_init(&cond, NULL);
    threads[num_threads];
    // pthread_create(&one_thread, NULL, &startThread, NULL);
    for(int i = 0; i < num_threads; i++){
        pthread_create(&threads[i], NULL, &startThread, NULL);
    }
    return ;
}   

void async_run(void (*hanlder)(int), int args) {
    
    // hanlder(args);

    my_item_t *it_pt = malloc(sizeof(my_item_t));
    it_pt->taskFunction = hanlder;
    it_pt->args = args;

    pthread_mutex_lock(&mutex);
    printf("done...");
    DL_APPEND(work_q_pt->head, it_pt);
    work_q_pt->size++;
    printf("now queue size is %d", work_q_pt->size);
    pthread_mutex_unlock(&mutex);
    return ;
}
