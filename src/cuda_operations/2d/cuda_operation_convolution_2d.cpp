/**
* @file    2D Optical flow using NVIDIA CUDA
* @author  Institute for Photon Science and Synchrotron Radiation, Karlsruhe Institute of Technology
*
* @date    2015-2018
* @version 0.5.0
*
*
* @section LICENSE
*
* This program is copyrighted by the author and Institute for Photon Science and Synchrotron Radiation,
* Karlsruhe Institute of Technology, Karlsruhe, Germany;
*
* The current implemetation contains the following licenses:
*
* 1. TinyXml package:
*      Original code (2.0 and earlier )copyright (c) 2000-2006 Lee Thomason (www.grinninglizard.com). <www.sourceforge.net/projects/tinyxml>.
*      See src/utils/tinyxml.h for details.
*
*/

#include "cuda_operation_convolution_2d.h"

#include <cstring>
#include <iostream>
#include <cmath>

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector_types.h>

#include "src/data_types/data_structs.h"
#include "src/utils/common_utils.h"
#include "src/utils/cuda_utils.h"

using namespace std;

CudaOperationConvolution2D::CudaOperationConvolution2D()
  : CudaOperationBase("CUDA Convolution 2D")
{
}

bool CudaOperationConvolution2D::Initialize(const OperationParameters* params)
{
  initialized_ = false;
  
  if (!params) {
    std::printf("Operation: '%s'. Initialization parameters are missing.\n", GetName());
    return initialized_;
  }

  DataSize3 container_size;
  GET_PARAM_OR_RETURN_VALUE(*params, DataSize3, container_size, "container_size", initialized_);

  dev_container_size_ = container_size;
 
  char exec_path[256];
  Utils::GetExecutablePath(exec_path, 256);
  std::strcat(exec_path, "/kernels/convolution_2d.ptx");

  if (!CheckCudaError(cuModuleLoad(&cu_module_, exec_path))) {
    if (!CheckCudaError(cuModuleGetFunction(&cuf_convolution_rows_, cu_module_, "convolutionRowsKernel")) &&
        !CheckCudaError(cuModuleGetFunction(&cuf_convolution_cols_, cu_module_, "convolutionColumnsKernel"))) {
      size_t const_size;

      /* Get the pointer to the constant memory and copy data */
      if (!CheckCudaError(cuModuleGetGlobal(&dev_constants_, &const_size, cu_module_, "container_size"))) {
        if (const_size == sizeof(container_size)) {
          if (!CheckCudaError(cuMemcpyHtoD(dev_constants_, &container_size, sizeof(container_size)))) {
            initialized_ = true;
          }
        }
      }

    } else {
      CheckCudaError(cuModuleUnload(cu_module_));
      cu_module_ = nullptr;
    }
  }
  return initialized_;
}

void CudaOperationConvolution2D::ComputeGaussianKernel(float sigma, size_t precision, float pixel_size)
{
    kernel_radius_ = (size_t)(precision * sigma / pixel_size);

    kernel_ = new float[2*kernel_radius_ + 1];
    kernel_length_ = 2*kernel_radius_ + 1;

    int r = static_cast<int>(kernel_radius_);

    for (int i = -r; i <= r; i++) {
        float val = 1.0 / (sigma * std::sqrt(2.0 * 3.1415926)) * std::exp(-(i * i * pixel_size * pixel_size) / (2.0 * sigma * sigma));
        kernel_[i+r] = val;
    }

    // Normalize convolution mask vector 
    float sum = 0.0;

    for (unsigned int i = 0; i < kernel_length_; i++) {
        sum = sum + kernel_[i];
    }

    for (unsigned int i = 0; i < kernel_length_; i++) {
        kernel_[i] = kernel_[i] / sum;
    }

   /* for (unsigned int i = 0; i < kernel_length_; i++) {
        kernel_[i] = 1.0;
    }*/

}

void CudaOperationConvolution2D::PrintConvolutionKernel()
{
    if (kernel_ == nullptr)
        std::printf("Error: Convolution kernel is not initialized.\n");
    else {
        std::printf("Convolution kernel (radius = %d)\n", kernel_radius_);

        for (unsigned int i = 0; i < kernel_length_; i++) {
            std::printf("%.4f ", kernel_[i]);
        }
        std::printf("\n\n");

    }

}



void CudaOperationConvolution2D::Execute(OperationParameters& params)
{  
  if (!IsInitialized()) {
    return;
  }

  CUdeviceptr dev_input = 0;
  CUdeviceptr dev_output = 0;
  CUdeviceptr dev_temp = 0;
  DataSize3 data_size;
  float gaussian_sigma;

  GET_PARAM_OR_RETURN(params, CUdeviceptr, dev_input,  "dev_input");
  GET_PARAM_OR_RETURN(params, CUdeviceptr, dev_output, "dev_output");
  GET_PARAM_OR_RETURN(params, CUdeviceptr, dev_temp, "dev_temp");
  GET_PARAM_OR_RETURN(params, DataSize3, data_size,    "data_size");
  GET_PARAM_OR_RETURN(params, float, gaussian_sigma, "gaussian_sigma");


  if (dev_input == dev_output) {
    std::printf("Operation '%s': Error. Input buffer cannot serve as output buffer.", GetName());
    return;
  }
    
    // TODO: Make computation of kernel once for all images
    ComputeGaussianKernel(gaussian_sigma, 3, 1.0);
    //PrintConvolutionKernel();

    size_t const_size;

    /* Get the pointer to the constant memory and copy data */
    CheckCudaError(cuModuleGetGlobal(&dev_constants_, &const_size, cu_module_, "c_Kernel"));
    CheckCudaError(cuMemcpyHtoD(dev_constants_, kernel_, kernel_length_*sizeof(float)));

    int pitch = static_cast<int>(dev_container_size_.pitch / sizeof(float));

    /* 1. Process in horizontal direction (rows) */
    ResampleRows(dev_input, dev_temp, data_size, pitch);


    /* 2. Process in vertical direction (columns) */
    ResampleColumns(dev_temp, dev_output, data_size, pitch);


}

void CudaOperationConvolution2D::ResampleRows(CUdeviceptr input, CUdeviceptr output, DataSize3& data_size, size_t pitch) const
{
    int kernel_radius = static_cast<int>(kernel_radius_);

    unsigned int rows_results_steps = 4;
    unsigned int rows_halo_steps = 1;

    dim3 row_block_dim ={ 16, 4 };
    dim3 row_grid_dim ={ static_cast<unsigned int>((data_size.width  + (row_block_dim.x * rows_results_steps) - 1) / (row_block_dim.x * rows_results_steps)),
        static_cast<unsigned int>((data_size.height + row_block_dim.y - 1) / row_block_dim.y) };

    /* Calculate the needed shared memory size */
    int shared_memory_size;
    CUdevice cu_device;
    CheckCudaError(cuDeviceGet(&cu_device, 0));
    CheckCudaError(cuDeviceGetAttribute(&shared_memory_size, CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK, cu_device));

    int needed_shared_memory_size =
        ((rows_results_steps + 2*rows_halo_steps)*row_block_dim.x) * (row_block_dim.y) * sizeof(float);

    //std::printf("Shared memory: %d Needed: %d \n", shared_memory_size, needed_shared_memory_size);

    if (needed_shared_memory_size > shared_memory_size) {
        std::printf("<%s>: Error shared memory allocation. Reduce the thread block size.\n", GetName());
        std::printf("Shared memory: %d Needed: %d \n", shared_memory_size, needed_shared_memory_size);
        return;
    }

    /*std::printf("Starting Convolution kernel: Grid: %d x %d, Blocks %d x %d \n", row_grid_dim.x,
        row_grid_dim.y,
        row_block_dim.x,
        row_block_dim.y);*/


    void* args[6] ={
        &output,
        &input,
        &data_size.width,
        &data_size.height,
        &pitch,
        &kernel_radius };

    CheckCudaError(cuLaunchKernel(cuf_convolution_rows_,
        row_grid_dim.x, row_grid_dim.y, row_grid_dim.z,
        row_block_dim.x, row_block_dim.y, row_block_dim.z,
        needed_shared_memory_size,
        NULL,
        args,
        NULL));
}

void CudaOperationConvolution2D::ResampleColumns(CUdeviceptr input, CUdeviceptr output, DataSize3& data_size, size_t pitch) const
{
    int kernel_radius = static_cast<int>(kernel_radius_);

    unsigned int cols_results_steps = 4;
    unsigned int cols_halo_steps = 1;

    dim3 col_block_dim ={ 4, 16 };
    dim3 col_grid_dim ={ static_cast<unsigned int>((data_size.width  + col_block_dim.x - 1) / col_block_dim.x),
        static_cast<unsigned int>((data_size.height + (col_block_dim.y * cols_results_steps) - 1) / (col_block_dim.y * cols_results_steps)) };

    int shared_memory_size;
    CUdevice cu_device;
    CheckCudaError(cuDeviceGet(&cu_device, 0));
    CheckCudaError(cuDeviceGetAttribute(&shared_memory_size, CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK, cu_device));

    int needed_shared_memory_size =
        (col_block_dim.x * ((cols_results_steps + 2*cols_halo_steps)*col_block_dim.y + 0)) * sizeof(float);

    //std::printf("Shared memory: %d Needed: %d \n", shared_memory_size, needed_shared_memory_size);

    if (needed_shared_memory_size > shared_memory_size) {
        std::printf("<%s>: Error shared memory allocation. Reduce the thread block size.\n", GetName());
        std::printf("Shared memory: %d Needed: %d \n", shared_memory_size, needed_shared_memory_size);
        return;
    }

   /* std::printf("Starting Convolution kernel: Grid: %d x %d, Blocks %d x %d \n", col_grid_dim.x,
        col_grid_dim.y,
        col_block_dim.x,
        col_block_dim.y);
*/


    void* args[6] ={
        &output,
        &input,
        &data_size.width,
        &data_size.height,
        &pitch,
        &kernel_radius };

    CheckCudaError(cuLaunchKernel(cuf_convolution_cols_,
        col_grid_dim.x, col_grid_dim.y, col_grid_dim.z,
        col_block_dim.x, col_block_dim.y, col_block_dim.z,
        needed_shared_memory_size,
        NULL,
        args,
        NULL));
}




CudaOperationConvolution2D::~CudaOperationConvolution2D()
{
    if (kernel_ != nullptr)
        delete kernel_;

}