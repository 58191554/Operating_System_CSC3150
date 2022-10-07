obj-m	:= hello.o
KVERSION := $(shell uname -r)
PWD	:= $(shell pwd)

all:
	$(MAKE) -C /lib/modules/$(KVERSION)/build M=$(PWD) modules
clean:
	rm -rf *.o *.ko *.mod.* *.cmd *.symvers *.order *.mod
