#ifndef __ASYNC__
#define __ASYNC__

#include <pthread.h>


// the task struct
typedef struct my_item {

    struct my_item *next;
    struct my_item *prev;
    void (*taskFunction)(int);
    int args;

} my_item_t;

typedef struct my_queue {

    int size;
    my_item_t *head;
    my_item_t *tail;

} my_queue_t;

my_queue_t *work_q_pt;
pthread_t one_thread;
pthread_t threads[];
pthread_mutex_t mutex;\
pthread_cond_t cond;

void enqueue(my_queue_t *queue, my_item_t *item);
my_item_t * dequeue(my_queue_t *queue);
void async_init(int);
void async_run(void (*fx)(int), int args);


#endif
