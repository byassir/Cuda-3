################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/greyScale.cu 

CU_DEPS += \
./src/greyScale.d 

OBJS += \
./src/greyScale.o 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc -I/usr/include/opencv -I/usr/include/opencv2 -I/usr/local/cuda/include -include /usr/include/opencv2/opencv.hpp -G -g -O0 -gencode arch=compute_30,code=sm_30 -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	nvcc --compile -G -I/usr/include/opencv -I/usr/include/opencv2 -I/usr/local/cuda/include -O0 -g -gencode arch=compute_30,code=compute_30 -gencode arch=compute_30,code=sm_30 -include /usr/include/opencv2/opencv.hpp  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


