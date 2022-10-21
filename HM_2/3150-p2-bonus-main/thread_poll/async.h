#ifndef __ASYNC__
#define __ASYNC__

#include <pthread.h>

// the task struct
typedef struct my_item {

    struct my_item *next;
    struct my_item *prev;
    /* TODO */
    void (*taskFunction)(int x);
    int args;

} my_item_t;

typedef struct my_queue {

    int size;
    my_item_t *head;
    /* TODO */
} my_queue_t;

void async_init(int);
void async_run(void (*fx)(int), int args);


#endif
