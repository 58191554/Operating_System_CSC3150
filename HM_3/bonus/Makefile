build:
	@nvcc -g -G --relocatable-device-code=true main.cu user_program.cu virtual_memory.cu -o test

run: build
	@./test
