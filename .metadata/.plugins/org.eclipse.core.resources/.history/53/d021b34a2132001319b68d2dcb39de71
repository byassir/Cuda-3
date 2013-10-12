################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/loadSaveImage.cpp \
../src/reference_calc.cpp 

CU_SRCS += \
../src/toneMap.cu 

CU_DEPS += \
./src/toneMap.d 

OBJS += \
./src/loadSaveImage.o \
./src/reference_calc.o \
./src/toneMap.o 

CPP_DEPS += \
./src/loadSaveImage.d \
./src/reference_calc.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc -I/usr/local/include -I/usr/local/cuda-5.0/samples -I/usr/include/opencv -I/usr/local/cuda/include -G -g -O0 -gencode arch=compute_30,code=sm_30 -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	nvcc -I/usr/local/include -I/usr/local/cuda-5.0/samples -I/usr/include/opencv -I/usr/local/cuda/include -G -g -O0 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/%.o: ../src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc -I/usr/local/include -I/usr/local/cuda-5.0/samples -I/usr/include/opencv -I/usr/local/cuda/include -G -g -O0 -gencode arch=compute_30,code=sm_30 -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	nvcc --compile -G -I/usr/local/include -I/usr/local/cuda-5.0/samples -I/usr/include/opencv -I/usr/local/cuda/include -O0 -g -gencode arch=compute_30,code=compute_30 -gencode arch=compute_30,code=sm_30  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


