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

#include "cuda_operation_correlation_2d.h"

#include <cstring>
#include <iostream>

#include <cuda.h>
#include <vector_types.h>

#include "src/data_types/data_structs.h"
#include "src/utils/common_utils.h"
#include "src/utils/cuda_utils.h"

using namespace std;

CudaOperationCorrelation2D::CudaOperationCorrelation2D()
  : CudaOperationBase("CUDA Correlation 2D")
{
}

bool CudaOperationCorrelation2D::Initialize(const OperationParameters* params)
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
  std::strcat(exec_path, "/kernels/correlation_2d.ptx");

  if (!CheckCudaError(cuModuleLoad(&cu_module_, exec_path))) {

    if (!CheckCudaError(cuModuleGetFunction(&cuf_correlation_, cu_module_, "correlation_2d")) &&
        !CheckCudaError(cuModuleGetFunction(&cuf_find_peak_, cu_module_, "find_peak_2d")) &&
        !CheckCudaError(cuModuleGetFunction(&cuf_select_peak_, cu_module_, "select_peak_2d")) ) {
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

void CudaOperationCorrelation2D::Execute(OperationParameters& params)
{
  if (!IsInitialized()) {
    return;
  }

  CUdeviceptr dev_image;
  CUdeviceptr dev_flow_x;
  CUdeviceptr dev_flow_y;
  CUdeviceptr dev_corr;

  CUdeviceptr dev_corr_ext;
  CUdeviceptr dev_corr_max_ext;

  DataSize3 data_size;

  size_t corr_window_size;
  size_t extended_pitch_size;


  GET_PARAM_OR_RETURN(params, CUdeviceptr, dev_image,  "image");
  GET_PARAM_OR_RETURN(params, CUdeviceptr, dev_flow_x, "flow_x");
  GET_PARAM_OR_RETURN(params, CUdeviceptr, dev_flow_y, "flow_y");
  GET_PARAM_OR_RETURN(params, CUdeviceptr, dev_corr, "corr");

  GET_PARAM_OR_RETURN(params, CUdeviceptr, dev_corr_ext, "corr_ext");
  GET_PARAM_OR_RETURN(params, CUdeviceptr, dev_corr_max_ext, "corr_max_ext");

  GET_PARAM_OR_RETURN(params, DataSize3, data_size, "data_size");
  GET_PARAM_OR_RETURN(params, size_t, corr_window_size, "corr_window_size");
  GET_PARAM_OR_RETURN(params, size_t, extended_pitch_size, "pitch_ext");



    /* Be careful with the thread block size. Thread block size in each dimension
        should be greater than radius / 2, because radius / 2 threads are used to 
        load halo around the thread block. */
    //dim3 block_dim ={ 8, 8};
    //dim3 block_dim ={ static_cast<unsigned int>(corr_window_size), static_cast<unsigned int>(corr_window_size) };

    dim3 block_dim ={ 16, 16 };

    dim3 grid_dim ={ static_cast<unsigned int>((data_size.width  + block_dim.x - 1) / block_dim.x),
                        static_cast<unsigned int>((data_size.height + block_dim.y - 1) / block_dim.y)};


    /* Calculate the needed shared memory size */
    int shared_memory_size;
    CUdevice cu_device;
    CheckCudaError(cuDeviceGet(&cu_device, 0));
    CheckCudaError(cuDeviceGetAttribute(&shared_memory_size, CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK, cu_device));


    /*-------------------------------------------*/
    /* Step 1: Compute correlation*/
    /*-------------------------------------------*/

    //int radius_2 = block_dim.x / 2;
    int radius_2 = corr_window_size / 2;

    int needed_shared_memory_size =
        (block_dim.x + 2 * radius_2) * (block_dim.y + 2 * radius_2) * sizeof(float);

    std::printf("Shared memory: %d Needed: %d \n", shared_memory_size, needed_shared_memory_size);

    if (needed_shared_memory_size > shared_memory_size) {
        std::printf("<%s>: Error shared memory allocation. Reduce the thread block size.\n", GetName());
        std::printf("Shared memory: %d Needed: %d \n", shared_memory_size, needed_shared_memory_size);
        return;
    }

    std::cout<<"Pitch: "<<extended_pitch_size<<std::endl;

    
    void* args[6] = { 
        &dev_image,
        &data_size.width,
        &data_size.height,
        &corr_window_size,
        &extended_pitch_size,
        &dev_corr_ext };

    CheckCudaError(cuLaunchKernel(cuf_correlation_,
                                    grid_dim.x, grid_dim.y, grid_dim.z,
                                    block_dim.x, block_dim.y, block_dim.z,
                                    needed_shared_memory_size,
                                    NULL,
                                    args,
                                    NULL));

    

    //CheckCudaError(cuStreamSynchronize(NULL));

    
    /*-------------------------------------------*/
    /* Step 2: Find peaks*/
    /*-------------------------------------------*/
    
    size_t min_distance = 1;

    size_t ext_width = data_size.width*corr_window_size;
    size_t ext_height = data_size.height*corr_window_size;

    block_dim ={ 16, 16 };

    grid_dim ={ static_cast<unsigned int>((ext_width  + block_dim.x - 1) / block_dim.x),
        static_cast<unsigned int>((ext_height + block_dim.y - 1) / block_dim.y) };

    cout<<"Grid dim:"<<grid_dim.x<<" "<<grid_dim.y<<endl;

    needed_shared_memory_size =
        (block_dim.x + 2 * min_distance) * (block_dim.y + 2 * min_distance) * sizeof(float);

    cout<<"Width:"<<ext_width<<endl;
    cout<<"Height:"<<ext_height<<endl;

    void* args2[7] ={
        &dev_corr_ext,
        &ext_width,
        &ext_height,
        &corr_window_size,
        &min_distance,
        &extended_pitch_size,
        &dev_corr_max_ext };

    CheckCudaError(cuLaunchKernel(cuf_find_peak_,
        grid_dim.x, grid_dim.y, grid_dim.z,
        block_dim.x, block_dim.y, block_dim.z,
        needed_shared_memory_size,
        NULL,
        args2,
        NULL));


    //CheckCudaError(cuStreamSynchronize(NULL));

    


    /*---------------------------------------------------------------*/
    /* Step 3: Select peaks and corresponding velocity components*/
    /*--------------------------------------------------------------*/

    block_dim ={ 16, 16 };

    grid_dim ={ static_cast<unsigned int>((data_size.width  + block_dim.x - 1) / block_dim.x),
        static_cast<unsigned int>((data_size.height + block_dim.y - 1) / block_dim.y) };

    needed_shared_memory_size = 0;


    void* args3[8] ={
        &dev_corr_max_ext,
        &data_size.width,
        &data_size.height,
        &corr_window_size,
        &extended_pitch_size,
        &dev_flow_x,
        &dev_flow_y,
        &dev_corr};

    CheckCudaError(cuLaunchKernel(cuf_select_peak_,
        grid_dim.x, grid_dim.y, grid_dim.z,
        block_dim.x, block_dim.y, block_dim.z,
        needed_shared_memory_size,
        NULL,
        args3,
        NULL));

    

}