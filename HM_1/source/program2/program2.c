#include <linux/err.h>
#include <linux/fs.h>
#include <linux/jiffies.h>
#include <linux/kernel.h>
#include <linux/kmod.h>
#include <linux/kthread.h>
#include <linux/module.h>
#include <linux/pid.h>
#include <linux/printk.h>
#include <linux/sched.h>
#include <linux/slab.h>

MODULE_LICENSE("GPL");

// copy from <waitstatus.h> the GNU C Library. Copyright (C) 1996-2020 Free
// Software Foundation, Inc.
/* If WIFEXITED(STATUS), the low-order 8 bits of the status.  */
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

const char *sigprompt[] = {"INVALID",
                           "hung up",
                           "interupted",
                           "quitted",
                           "stopped by ill-formed instruction",
                           "trapped",
                           "aborted",
                           "exited by bus error",
                           "exited by computation error",
                           "killed",
                           "exited by user defined signal",
                           "exited by segmentation fault",
                           "exited by user defined signal",
                           "piped",
                           "alarmed",
                           "terminated",
                           "a stack fault",
                           "received a SIGCHLD signal",
                           "received a coninue signal"};

const char *const signame[] = {
    "INVALID", "SIGHUP",  "SIGINT",    "SIGQUIT", "SIGILL",    "SIGTRAP",
    "SIGABRT", "SIGBUS",  "SIGFPE",    "SIGKILL", "SIGUSR1",   "SIGSEGV",
    "SIGUSR2", "SIGPIPE", "SIGALRM",   "SIGTERM", "SIGSTKFLT", "SIGCHLD",
    "SIGCONT", "SIGSTOP", "SIGTSTP",   "SIGTTIN", "SIGTTOU",   "SIGURG",
    "SIGXCPU", "SIGXFSZ", "SIGVTALRM", "SIGPROF", "SIGWINCH",  "SIGPOLL",
    "SIGPWR",  "SIGSYS",  NULL};

static struct task_struct *task;
int status;
int sig;

// extern essential linux kernel stuff
extern pid_t kernel_clone(struct kernel_clone_args *args);
extern int do_execve(struct filename *filename,
                     const char __user *const __user *__argv,
                     const char __user *const __user *__envp);
extern struct filename *getname_kernel(const char *filename);
extern long do_wait(struct wait_opts *wo);

// print out the signal in the kernel log
void output_info(int status) {

  if (__WIFSIGNALED(status)) {
    sig = __WTERMSIG(status);
    printk("[program2] : get %s signal\n", signame[sig]);
    if (sig <= 18)
      printk("[program2] : child process is %s.\n", sigprompt[sig]);
    else
      printk("[program2] : the return signal is %d\n", sig);
    printk("[program2] : the return signal is %d\n", sig);
  }

  if (__WIFSTOPPED(status)) {
    sig = __WSTOPSIG(status);
    printk("[program2] : child process get %s signal\n", signame[sig]);
    if (sig <= 18)
      printk("[program2] : child process is %s.\n", sigprompt[sig]);
    else
      printk("[program2] : the return signal is %d\n", sig);
  }

  if (__WIFEXITED(status)) {
    printk("[program2] : Normal termination with EXIT STATUS = %d\n",
           __WEXITSTATUS(status));
  }
}

// implement execute function
int my_exec(void *argc) {
  int result;
  const char path[] = "/tmp/test";
  // const char path[] = "/home/vagrant/CSC3150/HM_1/source/program1/trap";
  printk("[program2] : child process");
  return do_execve(getname_kernel(path), NULL, NULL);
}

// implement my_wait function
void my_wait(struct task_struct *pid) {

  struct wait_opts wo;
  struct pid *wo_pid = NULL;
  enum pid_type type;
  type = PIDTYPE_PID;
  wo_pid = find_get_pid(pid);

  wo.wo_type = type;
  wo.wo_pid = wo_pid;
  wo.wo_flags = WEXITED;
  wo.wo_info = NULL;
  wo.wo_stat = (int __user *)&status;
  wo.wo_rusage = NULL;

  int a;
  a = do_wait(&wo);

  printk("[program2] : wo_stat = %d\n", wo.wo_stat);

  output_info(wo.wo_stat);

  put_pid(wo_pid);
  return;
}

// implement fork function
int my_fork(void *argc) {

  // set default sigaction for current process
  int i;
  printk("[program2] : module_init kthread start");
  struct k_sigaction *k_action = &current->sighand->action[0];
  for (i = 0; i < _NSIG; i++) {
    k_action->sa.sa_handler = SIG_DFL;
    k_action->sa.sa_flags = 0;
    k_action->sa.sa_restorer = NULL;
    sigemptyset(&k_action->sa.sa_mask);
    k_action++;
  }

  /* fork a process using kernel_clone or kernel_thread */

  struct kernel_clone_args kargs = {
      .flags = SIGCHLD,
      .exit_signal = SIGCHLD,
      .stack = &my_exec,
      .stack_size = 0,
      .parent_tid = NULL,
      .child_tid = NULL,
  };
  pid_t pid = kernel_clone(&kargs);

  printk("[program2] : The child process has pid = %d\n", (int)pid);
  printk("[program2] : This is the parent process, pid = %d\n", task->pid);

  /* wait until child process terminates */
  my_wait(pid);

  return 0;
}

static int __init program2_init(void) {

  printk("[program2] : Module_init {name: TONG ZHEN} {id: 120090694}\n");

  /* write your code here */

  /* create a kernel thread to run my_fork */
  task = kthread_create(&my_fork, NULL, "my_thread");

  /* execute a test program in child process */
  if (!IS_ERR(task)) {
    printk("[program2] : module_init create kthread start");
    // throw the task from new mode to task_list to run

    wake_up_process(task);
  }
  return 0;
}

static void __exit program2_exit(void) { printk("[program2] : Module_exit\n"); }

module_init(program2_init);
module_exit(program2_exit);
