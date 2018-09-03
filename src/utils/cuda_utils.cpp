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

#include "cuda_utils.h"

#include <cstring>

bool InitCudaContextWithFirstAvailableDevice(CUcontext* cu_context)
{
  if (CheckCudaError(cuInit(0))) {
    return false;
  }

  int cu_device_count;
  if (CheckCudaError(cuDeviceGetCount(&cu_device_count))) {
    return false;
  }
  CUdevice cu_device;

  if (cu_device_count == 0) {
    printf("There are no cuda capable devices.");
    return false;
  }

  if (CheckCudaError(cuDeviceGet(&cu_device, 0))) {
    return false;
  }

  char cu_device_name[64];
  if (CheckCudaError(cuDeviceGetName(cu_device_name, 64, cu_device))) {
    return false;
  }

  int launch_timeout;
  CheckCudaError(cuDeviceGetAttribute(&launch_timeout, CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT, cu_device));

  printf("CUDA Device: %s. Launch timeout: %s\n", cu_device_name, (launch_timeout ? "Yes" : "No"));

  if (CheckCudaError(cuCtxCreate(cu_context, 0, cu_device))) {
    return false;
  }

  return true;
}



void CopyData2DtoDevice(Data2D& data2d, CUdeviceptr device_ptr, size_t device_height, size_t device_pitch)
{
    CUDA_MEMCPY2D cu_copy2d;
    std::memset(&cu_copy2d, 0, sizeof(CUDA_MEMCPY2D));

    cu_copy2d.srcMemoryType = CU_MEMORYTYPE_HOST;
    cu_copy2d.srcHost = data2d.DataPtr();
    cu_copy2d.srcPitch = data2d.Width() * sizeof(float);
         
    cu_copy2d.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    cu_copy2d.dstDevice = device_ptr;
    cu_copy2d.dstPitch = device_pitch;
          
    cu_copy2d.WidthInBytes = data2d.Width() * sizeof(float);
    cu_copy2d.Height = data2d.Height();

    CheckCudaError(cuMemcpy2D(&cu_copy2d));
}


void CopyData2DFromDevice(CUdeviceptr device_ptr, Data2D& data2d, size_t device_height, size_t device_pitch)
{
    CUDA_MEMCPY2D cu_copy2d;
    std::memset(&cu_copy2d, 0, sizeof(CUDA_MEMCPY2D));

    cu_copy2d.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    cu_copy2d.srcDevice = device_ptr;
    cu_copy2d.srcPitch = device_pitch;

         
    cu_copy2d.dstMemoryType = CU_MEMORYTYPE_HOST;
    cu_copy2d.dstHost = data2d.DataPtr();
    cu_copy2d.dstPitch = data2d.Width() * sizeof(float);

       
    cu_copy2d.WidthInBytes = data2d.Width() * sizeof(float);
    cu_copy2d.Height = data2d.Height();

    CheckCudaError(cuMemcpy2D(&cu_copy2d));
}
