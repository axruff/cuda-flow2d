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

#ifndef GPUFLOW3D_UTILS_CUDA_UTILS_H_
#define GPUFLOW3D_UTILS_CUDA_UTILS_H_

#include <cstdio>

#include <cuda.h>

#include "src/data_types/data2d.h"

#define CheckCudaError(error) __CheckCudaError (error, __FILE__, __LINE__)

inline bool __CheckCudaError(CUresult error, const char* file, const int line)
{
  if (error != CUDA_SUCCESS)
  {
    const char *error_name = nullptr;
    const char *error_string = nullptr;

//#define NEW_CUDA_DRIVER
#ifdef NEW_CUDA_DRIVER
    cuGetErrorName(error, &error_name);
    cuGetErrorString(error, &error_string);
#endif

    std::fprintf(stderr, "Driver API error = %04d %s\n%s\n from file <%s>, line %i.\n",
      error, error_name, error_string, file, line);
    return true;
  }
  return false;
}

bool InitCudaContextWithFirstAvailableDevice(CUcontext* cu_context);

void CopyData2DtoDevice(Data2D& data2d, CUdeviceptr device_ptr, size_t device_height, size_t device_pitch);
void CopyData2DFromDevice(CUdeviceptr device_ptr, Data2D& data2d, size_t device_height, size_t device_pitch);



#endif // !GPUFLOW3D_UTILS_CUDA_UTILS_H_
