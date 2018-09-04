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

#define __CUDACC__
#include <math_functions.h>

#include "src/data_types/data_structs.h"

//#define IND(X, Y, Z) (((Z) * container_size.height + (Y)) * (container_size.pitch / sizeof(float)) + (X)) 
#define IND(X, Y) ((Y) * (container_size.pitch / sizeof(float)) + (X)) 

__constant__ DataSize3 container_size;

extern "C" __global__ void registration_2d(
  const float* frame_0,
  const float* frame_1,
  const float* flow_u,
  const float* flow_v,
        size_t width,
        size_t height,
        float  hx,
        float  hy,
        float* output)
{
  dim3 global_id(blockDim.x * blockIdx.x + threadIdx.x,
                blockDim.y * blockIdx.y + threadIdx.y);

  if (global_id.x < width && global_id.y < height) {
    float x_f = global_id.x + (flow_u[IND(global_id.x, global_id.y)] * (1.f / hx));
    float y_f = global_id.y + (flow_v[IND(global_id.x, global_id.y)] * (1.f / hy));

    if ((x_f < 0.) || (x_f > width - 1) || (y_f < 0.) || (y_f > height - 1) ||  isnan(x_f) || isnan(y_f)) { 
      output[IND(global_id.x, global_id.y)] = frame_0[IND(global_id.x, global_id.y)];

    } else {
      int x = (int) floorf(x_f); 
      int y = (int) floorf(y_f); 
      float delta_x = x_f - (float) x;
      float delta_y = y_f - (float) y;

      int x_1 = min(int(width -1), x + 1);
      int y_1 = min(int(height - 1), y + 1);

      float value =
        (1.f - delta_x) * (1.f - delta_y) * frame_1[IND(x  , y )] +
        (      delta_x) * (1.f - delta_y) * frame_1[IND(x_1, y )] +
        (1.f - delta_x) * (      delta_y) * frame_1[IND(x  , y_1)] +
        (      delta_x) * (      delta_y) * frame_1[IND(x_1, y_1)];


      output[IND(global_id.x, global_id.y)] = value;
    }
  }
}