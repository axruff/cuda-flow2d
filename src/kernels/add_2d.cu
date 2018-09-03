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

#include <device_launch_parameters.h>
#include <vector_types.h>

#include "src/data_types/data_structs.h"

//#define IND(X, Y, Z) (((Z) * container_size.height + (Y)) * (container_size.pitch / sizeof(float)) + (X)) 
#define IND(X, Y) ((Y) * (container_size.pitch / sizeof(float)) + (X)) 


__constant__ DataSize3 container_size;

extern "C" __global__ void add_2d(
        float* operand_0,
  const float* operand_1,
        size_t width,
        size_t height)
{
  dim3 global_id(blockDim.x * blockIdx.x + threadIdx.x,
                 blockDim.y * blockIdx.y + threadIdx.y);

  if (global_id.x < width && global_id.y < height) {
    operand_0[IND(global_id.x, global_id.y)] +=
      operand_1[IND(global_id.x, global_id.y)];
  }
}
