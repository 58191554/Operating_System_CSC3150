# CSC3150 Assignment 2
# HM 2 Threadings Report


---

## Environment

Ubuntu 20.04.4

![Untitled](HM%202%20Threadings%20Report%2042fe0a0722ea4a8c8410130771f8d134/Untitled.png)

Linux kernel version

![Untitled](HM%202%20Threadings%20Report%2042fe0a0722ea4a8c8410130771f8d134/Untitled%201.png)

GCC Version

![Untitled](HM%202%20Threadings%20Report%2042fe0a0722ea4a8c8410130771f8d134/Untitled%202.png)

## Frog Game

**Motivation:** This homework is a frog game. We are asked to use multithread to control the game. The user is allowed to use the keyboard to control the frog, while the game can run: moving the logs in the river and judging whether the frog dies.

**Detail Design:**

**Thread Design:** There will be two threads to control the game program, the thread receiving the keyboard signal, and the thread moving the logs. When the IO thread receives a keyboard signal, the mutex lock will block the logs-moving thread. Because the IO input is the user's reaction of the current game situation, we need to make sure the log position still be the one it used to be and output the new frog position graph. The logs-moving thread then continues to execute, while the IO thread is blocking.

**Game Status Design:** We designed the status check in the logs-moving thread. We define status 0 as normal execution, status 1 as a win, and status -1 as a lose. The logs-moving thread while loop iterates as long as the status is 0. It first gives the next state of log position and then checks if the frog is still in the feasible region. If it detects a lose position (droop into water or on the edge of the window), it will change the status to -1. If it detected a win position (on the other side of the bank), it will change the status to 1.

**Run:** 

```bash
$make
$./frog
```

**Output Picture:**

see Appendix 1

---

## Bonus: Thread Pool

**Motivation:** With coming tasks to execute, we create a queue to store those jobs, and create a number of threads to deal with them. When requests are given, some of the threads are executing the jobs, while there may have some spare threads waiting for coming work to do.

**Detail Design:** There will only be one thread accessing the queue and blocking other threads, either the asyn_run (main thread) adding new requests into the queue, or the threads that want to execute the task in the queue and dequeuing them.

The threads have only 3 states, doing dequeue, executing tasks, or being blocked by the mutex_lock. 

We used the <utilst.h> to implement  When a thread wants to execute a task, it first needs to wait for the queue to be not empty. If it is we use `pthread_cond_wait` instead of busy wait until the new task enqueue. When a new task is added to the queue, we call `pthread_cond_signal` to activate a wait and execute the task. Before the `pthread_cond_wait`

`pthread_cond_wait` refers to condition variables that are used with a `mutex`. The lock is acquired before calling `pthread_cond_wait`. The `pthread_cond_wait` function automatically releases the specified lock and waits for the condition variable to change. The specified mutex is automatically re-locked before the function call returns.

**Memory Usage:** Because the queue is created in the main thread, and the new tasks are added into the queue in `asyn_run` each time in stack memory. After the function has been done, the `my_item` pointer we add to the queue may disappear. Therefore, it `malloc` the `my_item` , and put the queue member pointer into heap memory.

**Run**

open a port in vscode

![Untitled](HM%202%20Threadings%20Report%2042fe0a0722ea4a8c8410130771f8d134/Untitled%203.png)

```bash
$make
$./httpserver --proxy inst.eecs.berkeley.edu:80 --port 8000 --num-threads 5
```

Open another terminal 

```bash
ab -n 5000 -c 10 http://127.0.0.1:8000/
```

**Output Picture:**

see Appendix 2

---

### Appendix

Appendix 1:

![Win case](HM%202%20Threadings%20Report%2042fe0a0722ea4a8c8410130771f8d134/Untitled%204.png)

Win case

![Lose case: fall into river](HM%202%20Threadings%20Report%2042fe0a0722ea4a8c8410130771f8d134/Untitled%205.png)

Lose case: fall into river

![Lose case: go out of window edge](HM%202%20Threadings%20Report%2042fe0a0722ea4a8c8410130771f8d134/Untitled%206.png)

Lose case: go out of window edge

![Quit game case](HM%202%20Threadings%20Report%2042fe0a0722ea4a8c8410130771f8d134/Untitled%207.png)

Quit game case

Appendix 2

![Clicking the website output](HM%202%20Threadings%20Report%2042fe0a0722ea4a8c8410130771f8d134/Untitled%208.png)

Clicking the website output

![ApacheBench output.](HM%202%20Threadings%20Report%2042fe0a0722ea4a8c8410130771f8d134/Untitled%209.png)

ApacheBench output.
