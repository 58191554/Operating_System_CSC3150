#include <assert.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define RPATH "/proc/"
#define SPATH "/stat"
#define STPATH "/status"
#define SN 100
#define LN 10000

struct Proc {
  unsigned int pid;
  char comm[LN];
  unsigned char state;
  unsigned int ppid;       // the number of the parent pid
  char name[SN];           // the name of the process
  int ccnt;                // the number of the child process
  struct Proc *chrilden[]; // child process as a array
};

struct Proc *ProcList[];
int ProcN = 0;

void set_proc(char *pid, struct Proc *proc) {

  char stat_path[SN] = {};
  char status_path[SN] = {};

  strcat(stat_path, RPATH);
  strcat(stat_path, pid);
  strcat(stat_path, SPATH);

  strcat(status_path, RPATH);
  strcat(status_path, pid);
  strcat(status_path, STPATH);

  FILE *fp = fopen(stat_path, "r");

  if (fp) {
    rewind(fp);
    fscanf(fp, "%d %s %c %d", &proc->pid, proc->comm, &proc->state,
           &proc->ppid);
    fclose(fp);
  } else {
    printf("Open file Error...\n");
  }

  fp = fopen(status_path, "r");
  char tmp[33];

  rewind(fp);
  fscanf(fp, "%s %s", tmp, proc->name);
  fclose(fp);
  proc->ccnt = 0;
}

void get_procs() {
  struct dirent *dirp;
  char dirname[SN] = RPATH;
  int i = 1;
  DIR *dp = opendir(dirname);

  while ((dirp = readdir(dp)) != NULL) {
    // d_type denotes the type, 4 for directory, 8 for regular file, and 0 for
    // unknown device
    if (dirp->d_type == 4) {
      // if the name start with a number check ascii
      if (48 <= dirp->d_name[0] && dirp->d_name[0] <= 57) {

        struct Proc *proc = (struct Proc *)malloc(sizeof(struct Proc));
        set_proc(dirp->d_name, proc);

        ProcList[i] = proc;
        i++;
      }
    }
  }
  ProcN = i;
  closedir(dp);
}

void gen_procs_tree(struct Proc *root) {

  for (int i = 1; i < ProcN; i++) {
    struct Proc *cp = ProcList[i];

    printf("%-15s %-15d %-15d %-15p\n", cp->name, cp->pid, cp->ppid, cp);

    for (int j = 0; j < ProcN; j++) {
      struct Proc *pp = ProcList[j];
      if (cp->ppid == pp->pid) {
        pp->chrilden[pp->ccnt] = cp;
        pp->ccnt++;
      }
    }
  }
}

void show_tree(struct Proc *node, int before_len) {
  printf("--%s(%d)", node->name, node->pid);
  // Adjust the INDENThere
  before_len += strlen(node->name) + 8;
  for (int i = 0; i < node->ccnt; i++) {

    if (i == 0) {
      printf("─┬─");
    } else if (i == node->ccnt - 1) {
      for (int j = 0; j < before_len; j++)
        printf(" ");
      printf("└─");
    } else {
      for (int j = 0; j < before_len; j++)
        printf(" ");
      printf("├─");
    }

    show_tree(node->chrilden[i], before_len);
    printf("\n");
  }
}

int main(int argc, char *argv[]) {
  for (int i = 0; i < argc; i++) {
    assert(argv[i]);
    printf("argv[%d] = %s\n", i, argv[i]);
  }
  assert(!argv[argc]);

  struct Proc *root = (struct Proc *)malloc(sizeof(struct Proc));

  strcpy(root->name, "root");
  root->pid = 0;
  root->ccnt = 0;
  ProcList[0] = root;

  get_procs();

  gen_procs_tree(root);

  show_tree(root, 0);

  return 0;
}
