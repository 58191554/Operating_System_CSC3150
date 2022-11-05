build:
	@nvcc -g -G --relocatable-device-code=true main.cu user_program.cu virtual_memory.cu -o main

run: build
	@./main
