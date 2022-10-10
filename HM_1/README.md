# HM1 Report System Process in Linux System

**Name: Tong Zhen** 

**id: 120090694**

---

## Environment Information

Linux Distribution: `Ubuntu 20.04` 
Linux Kernel Version: `5.10.146`
GCC Version: 9.4.0 (Ubuntu 9.4.0-1ubuntu1~20.04.1)

- **Linux kernel preparation**
    
    Step 1 Install a Linux kernel 5.10 from Tsinghua Source
    
    ```bash
    $wget https://mirror.tuna.tsinghua.edu.cn/kernel/v5.x/linux-5.10.27.tar.gz 
    # decompress the file
    $tar -xf linux-5.10.27.tar.gz
    ```
    
    Step 2 copy config file from /boot
    
    ```bash
    $cd /boot
    $cp config_file
    ```
    
    Step 3 get essential programs
    
    ```bash
    # to do menuconfig
    $apt-get install libncurses-dev
    $apt-get install flex
    $apt-get install bison
    $apt-get install libssl-dev
    $apt-get install libelf-dev
    $apt-get install 
    ```
    
    Step 4 compile 
    
    ```bash
    $make mrproper 
    $make clean 
    $make menuconfig
    
    $make bzImage -j$(nproc) 
    $make modules -j$(nproc) 
    $make –j$(nproc)
    
    $make modules_install
    $make install
    $reboot
    $uname -r
    ```
    
- **Export Symbol**
    
    linux-5.10.146/kernel/fork.c
    
    ```c
    2506: EXPORT_SYMBOL(kernel_clone);
    ```
    
    linux-5.10.146/fs/exec.c
    
    ```c
    2013: EXPORT_SYMBOL(do_execve);
    ```
    
    linux-5.10.146/fs/namei.c
    
    ```c
    212: EXPORT_SYMBOL(getname);
    250: EXPORT_SYMBOL(getname_kernel);
    ```
    
    linux-5.10.146/kernel/exit.c
    
    ```c
    1482: EXPORT_SYMBOL(do_wait);
    ```
    

## Task 1

### Fork a child process

When the program1 is running, we create a child process by calling `fork()` 

```c
pid_t pid = fork();
```

`pid_t` is the process descriptor, which is essentially an int：

- returns a negative number on failure
- returns two values on success: 0 and the child process ID

After we build the child process, the kernel will do the following things for us

1. Allocate new memory blocks and kernel data structures to the child process
2. Copy parts of the parent process's data structure (dataspace, stack, etc.) to the child process
3. Add the child process to the system process list
4. fork returns and then starts scheduling

### Execute program

Code begins from the fork function and is shared between parent and child, as it is executed by both parent and child. The child gets a copy of the parent's data space, heap, and stack. 

We identify the child process by the variable`pid == 0` Need to mention that, the child knows if its `pid` is 0 or not so that it knows if it was created successfully. We can also get the system's real pid by `getpid()`. In the child process, we can get file string data from the parent process memory and execute by calling `execute()` 

### Receive signal

In testing executable files compiled by c programs, different signals are returned to the parent process by calling `raise()` when meeting abnormal, and normal signal `SIGCHLD` will return by `exit(SIGCHLD)`

In the parent process, we use the `waitpid()` , because the parent process knows the pid of its child, it will assign the child status signal to `status` 

```c
int status;
waitpid(pid, &status, WUNTRACED);
```

We set the `WUNTRACED` because it is possible that the child will be killed/stopped.

Refered to the macros in <wait.h> we may get these signals.

```c
 1) SIGHUP       2) SIGINT       3) SIGQUIT      4) SIGILL       5) SIGTRAP
 6) SIGABRT      7) SIGBUS       8) SIGFPE       9) SIGKILL     10) SIGUSR1
11) SIGSEGV     12) SIGUSR2     13) SIGPIPE     14) SIGALRM     15) SIGTERM
16) SIGSTKFLT   17) SIGCHLD     18) SIGCONT     19) SIGSTOP     20) SIGTSTP
21) SIGTTIN     22) SIGTTOU     23) SIGURG      24) SIGXCPU     25) SIGXFSZ
26) SIGVTALRM   27) SIGPROF     28) SIGWINCH    29) SIGIO       30) SIGPWR
31) SIGSYS 
```

### Signal print

*Macros defined in <waitflags.h> can be used to analyze the status retu*

If the child process is normal, we check it by `WIFEXITED()`, and it will return true.

If the child process is stopped, we check it by `WIFSTOPPED()`, and it will return true.

If the child process is failed, we check it by `WIFSIGNALED()`, and it will return true.

Finally, whatever happened, we use the `exit(0)` to end the whole process.

### Program output

see [Appendix 1](https://www.notion.so/HM1-Report-System-Process-in-Linux-System-72852556d9f949068836ce6955347770)

## Task 2

### Create a kernel thread to run `my_fork()`

In the `program2_inti()` function, we need first to create a kernel thread, we use the `kthread_create(threadfn, data, namefmt, arg...)` The `threadfn` is my_fork(), the data is set as NULL, and the namefmt is “my_thread”. This create is a packaging for the `kernel_clone()`.

When the new task is created, we need to do the process by wake_up_process(task)

### Fork a child process in kernel mode

To fork a process using `kernel_clone()`, we first need to set the args in the `kernel_clone_args` struct. 

```c
struct kernel_clone_args kargs = {
      .flags = SIGCHLD,
      .exit_signal = SIGCHLD,
      .stack = &my_exec,
      .stack_size = 0,
      .parent_tid = NULL,
      .child_tid = NULL,
  };
  pid_t pid = kernel_clone(&kargs);
```

`my_exec()` is the function we want to do in the child process, and we wait the signal by using the `my_wait()` function.

```c
my_wait(pid);
```

### Execute the given test program

For convenience, we set the path of the test as 

```c
const char path[] = "/tmp/test";
```

We use the `do_execve()` function in linux kernel.

```c
do_execve(struct *filename, struct *argv, struct *envp)
```

 Need to mention that, if we want to get the filename sturct, `getname()` is not convenient, therefore we used the `getnamekernel()` in linux 5.10.x kernel.

Another thing worth to mention is that argv and envp are set to `NULL`.

### Wait for the child process and capture signal

We build the waiting function in `void my_wait(struct task_struct *pid)` , it takes in the child process pid.  Actually, what we do is just to capture the signal by creating a `wait_opts` struct and use the `do_wait(&wo)` in the linux kernel.

```c
struct wait_opts {
  enum pid_type wo_type;
  int wo_flags;
  struct pid *wo_pid;

  struct waitid_info *wo_info;
  int wo_stat;
  struct rusage *wo_rusage;

  wait_queue_entry_t child_wait;
  int notask_error;
};
```

Import argument here is the status, it will be sent in the do_wait() and later be assign to a signal that can be analyzed in the `output_info()` function.

### Signal analysis

In the `output_info()` function, we referred to the *<waitstatus.h> the GNU C Library*. It can do a transfermation.

```c
#define __WEXITSTATUS(status) (((status)&0xff00) >> 8)

/* If WIFSIGNALED(STATUS), the terminating signal.  */
#define __WTERMSIG(status) ((status)&0x7f)

/* If WIFSTOPPED(STATUS), the signal that stopped the child.  */
#define __WSTOPSIG(status) __WEXITSTATUS(status)

/* Nonzero if STATUS indicates normal termination.  */
#define __WIFEXITED(status) (__WTERMSIG(status) == 0)

/* Nonzero if STATUS indicates termination by a signal.  */
#define __WIFSIGNALED(status) (((signed char)(((status)&0x7f) + 1) >> 1) > 0)

/* Nonzero if STATUS indicates the child is stopped.  */
#define __WIFSTOPPED(status) (((status)&0xff) == 0x7f)
```

### Program output

see [Appendix 2](https://www.notion.so/HM1-Report-System-Process-in-Linux-System-72852556d9f949068836ce6955347770)

## Bonus

### Data structure for a process

Like a node in a tree, every process is with a pid number its parent and children. We define the `Proc`

```c
struct Proc {
    unsigned int pid; 
    char comm[LN];
    unsigned char state;
    unsigned int ppid;      // the number of the parent pid
    char name[SN];          // the name of the process
    int ccnt;               // the number of the child process
    struct Proc *chrilden[];    // child process as a array
};
```

### Visit process

we use the `get_proc()` function to access the /proc. 

```c
struct dirent {
      ino_t          d_ino;       /* inode number */
      off_t          d_off;       /* not an offset; see NOTES */
      unsigned short d_reclen;    /* length of this record */
      unsigned char  d_type;   /*type of file; not supported by all  filesystem types */              
                                                                                           
      char           d_name[256]; /* filename */
};
```

When we enter the /proc dirctory, we can view those process files

![Untitled](HM1%20Report%20System%20Process%20in%20Linux%20System%2072852556d9f949068836ce6955347770/Untitled.png)

After each use of `readdir()`, it will read the next file. readdir is to read all the files in the directory in sequence, one at a time. We always want the file open with a number.

### Set process information

When we get the right process file, use the `set_proc(char *pid, sturct Proc *proc)` we will set the set the path and name.

### Generate tree

Iterate the `Proc` in `ProcList` . In each iteration, iterate the `ProcList` again from the first element. we asign the child to the parent if **cp->ppid == pp->pid.**

```c
for (int i = 1; i < ProcN; i++) {
    struct Proc *cp = ProcList[i];

    for (int j = 0; j < ProcN; j++) {
      struct Proc *pp = ProcList[j];
      if **(cp->ppid == pp->pid)** {
        pp->chrilden[pp->ccnt] = cp;
        pp->ccnt++;
      }
    }
  }
```

### Show tree

We use dfs recursion to show the tree sturcture.

- when the child is the first in the child list, use─┬─to connect
- when the child is the middle in the child list, use├─to connect
- when the child is the last in the child list, use"└─to connect

### Output

see Appendix 3

## Run instruction

### Task 1

```bash
$make
$./program1 ./abort
$./task1.sh
```

### Task 2

```bash
$make
$sudo insmod program2.ko
$dmesg
$sudo rmmod program2.ko
```

### Bonus

```bash
$make
$./pstree
```

## Appendix

- Appendix 1
    
    ![Untitled](HM1%20Report%20System%20Process%20in%20Linux%20System%2072852556d9f949068836ce6955347770/Untitled%201.png)
    
    ![Untitled](HM1%20Report%20System%20Process%20in%20Linux%20System%2072852556d9f949068836ce6955347770/Untitled%202.png)
    
    ![Untitled](HM1%20Report%20System%20Process%20in%20Linux%20System%2072852556d9f949068836ce6955347770/Untitled%203.png)
    
    ![Untitled](HM1%20Report%20System%20Process%20in%20Linux%20System%2072852556d9f949068836ce6955347770/Untitled%204.png)
    
- [Appendix 2](https://www.notion.so/HM1-Report-System-Process-in-Linux-System-72852556d9f949068836ce6955347770)
    
    ![Untitled](HM1%20Report%20System%20Process%20in%20Linux%20System%2072852556d9f949068836ce6955347770/Untitled%205.png)
    
    ![Untitled](HM1%20Report%20System%20Process%20in%20Linux%20System%2072852556d9f949068836ce6955347770/Untitled%206.png)
    
    ![Untitled](HM1%20Report%20System%20Process%20in%20Linux%20System%2072852556d9f949068836ce6955347770/Untitled%207.png)
    
    ![Untitled](HM1%20Report%20System%20Process%20in%20Linux%20System%2072852556d9f949068836ce6955347770/Untitled%208.png)
    
    ![Untitled](HM1%20Report%20System%20Process%20in%20Linux%20System%2072852556d9f949068836ce6955347770/Untitled%209.png)
    
    ![Untitled](HM1%20Report%20System%20Process%20in%20Linux%20System%2072852556d9f949068836ce6955347770/Untitled%2010.png)
    
    ![Untitled](HM1%20Report%20System%20Process%20in%20Linux%20System%2072852556d9f949068836ce6955347770/Untitled%2011.png)
    
    ![Untitled](HM1%20Report%20System%20Process%20in%20Linux%20System%2072852556d9f949068836ce6955347770/Untitled%2012.png)
    
- Appendix 3
    
    ![Untitled](HM1%20Report%20System%20Process%20in%20Linux%20System%2072852556d9f949068836ce6955347770/Untitled%2013.png)
    
    ![Untitled](HM1%20Report%20System%20Process%20in%20Linux%20System%2072852556d9f949068836ce6955347770/Untitled%2014.png)
