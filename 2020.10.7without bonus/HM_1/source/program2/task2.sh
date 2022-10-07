make clean
make
sudo insmod program2.ko
dmesg
sleep 
sudo rmmod program2.ko
dmesg
