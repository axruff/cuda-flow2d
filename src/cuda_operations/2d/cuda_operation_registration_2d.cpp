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

#include "cuda_operation_registration_2d.h"

#include <cuda.h>
#include <vector_types.h>

#include "src/data_types/data_structs.h"
#include "src/utils/common_utils.h"
#include "src/utils/cuda_utils.h"

CudaOperationRegistration2D::CudaOperationRegistration2D()
  : CudaOperationBase("CUDA Registration 2D")
{
}

bool CudaOperationRegistration2D::Initialize(const OperationParameters* params)
{
  initialized_ = false;
  
  if (!params) {
    std::printf("Operation: '%s'. Initialization parameters are missing.\n", GetName());
    return initialized_;
  }

  DataSize3 container_size;
  GET_PARAM_OR_RETURN_VALUE(*params, DataSize3, container_size, "container_size", initialized_);
 
  char exec_path[256];
  Utils::GetExecutablePath(exec_path, 256);
  std::strcat(exec_path, "/kernels/registration_2d.ptx");

  if (!CheckCudaError(cuModuleLoad(&cu_module_, exec_path))) {
    if (!CheckCudaError(cuModuleGetFunction(&cuf_registration_, cu_module_, "registration_2d"))) {
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

void CudaOperationRegistration2D::Execute(OperationParameters& params)
{
  if (!IsInitialized()) {
    return;
  }

  CUdeviceptr dev_frame_0;
  CUdeviceptr dev_frame_1;
  CUdeviceptr dev_flow_u;
  CUdeviceptr dev_flow_v;
  CUdeviceptr dev_output;

  GET_PARAM_OR_RETURN(params, CUdeviceptr, dev_frame_0, "dev_frame_0");
  GET_PARAM_OR_RETURN(params, CUdeviceptr, dev_frame_1, "dev_frame_1");
  GET_PARAM_OR_RETURN(params, CUdeviceptr, dev_flow_u,  "dev_flow_u");
  GET_PARAM_OR_RETURN(params, CUdeviceptr, dev_flow_v,  "dev_flow_v");
  GET_PARAM_OR_RETURN(params, CUdeviceptr, dev_output,  "dev_output");

  float hx;
  float hy;
  DataSize3 data_size;

  GET_PARAM_OR_RETURN(params, float,     hx,        "hx");
  GET_PARAM_OR_RETURN(params, float,     hy,        "hy");
  GET_PARAM_OR_RETURN(params, DataSize3, data_size, "data_size");

  if (dev_frame_1 == dev_output) {
    std::printf("Operation '%s': Error. Input buffer cannot serve as output buffer.", GetName());
    return;
  }

  dim3 block_dim = { 16, 8, 1 };
  dim3 grid_dim = { static_cast<unsigned int>((data_size.width + block_dim.x - 1) / block_dim.x),
                    static_cast<unsigned int>((data_size.height + block_dim.y - 1) / block_dim.y),
                    1 };

  void* args[12] = { 
      &dev_frame_0,
      &dev_frame_1,
      &dev_flow_u,
      &dev_flow_v,
      &data_size.width,
      &data_size.height,
      &hx,
      &hy,
      &dev_output};

  CheckCudaError(cuLaunchKernel(cuf_registration_,
                                grid_dim.x, grid_dim.y, grid_dim.z,
                                block_dim.x, block_dim.y, block_dim.z,
                                0,
                                NULL,
                                args,
                                NULL));
}