build:
	@nvcc -g -G --relocatable-device-code=true main.cu user_program.cu file_system.cu -o main

run: build
	@./main