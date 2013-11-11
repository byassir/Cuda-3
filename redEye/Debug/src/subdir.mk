################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/loadSaveImage.cpp 

CU_SRCS += \
../src/redEye.cu 

CU_DEPS += \
./src/redEye.d 

OBJS += \
./src/loadSaveImage.o \
./src/redEye.o 

CPP_DEPS += \
./src/loadSaveImage.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc -I/usr/local/cuda -I/usr/local/include -I/usr/include/opencv -G -g -O0 -gencode arch=compute_30,code=sm_30 -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	nvcc -I/usr/local/cuda -I/usr/local/include -I/usr/include/opencv -G -g -O0 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/%.o: ../src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc -I/usr/local/cuda -I/usr/local/include -I/usr/include/opencv -G -g -O0 -gencode arch=compute_30,code=sm_30 -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	nvcc --compile -G -I/usr/local/cuda -I/usr/local/include -I/usr/include/opencv -O0 -g -gencode arch=compute_30,code=compute_30 -gencode arch=compute_30,code=sm_30  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


