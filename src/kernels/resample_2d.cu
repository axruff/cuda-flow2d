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

extern "C" __global__ void resample_x(
  const float* input,
        float* output,
        size_t out_width,
        size_t out_height,
        size_t in_width)
{
  dim3 globalID(blockDim.x * blockIdx.x + threadIdx.x,
                blockDim.y * blockIdx.y + threadIdx.y);

  if (globalID.x < out_width &&  globalID.y < out_height) {
    float delta = in_width / static_cast<float>(out_width);
    float normalization = out_width / static_cast<float>(in_width);

    float left_f = globalID.x * delta;
    float right_f = (globalID.x + 1) * delta;

    int left_i = static_cast<int>(floor(left_f));
    int right_i = min(in_width, static_cast<size_t>(ceil(right_f)));

    float value = 0.f;
    
    for (int j = 0; j < (right_i - left_i); j++) {
      float frac = 1.f;

      /* left boundary */
      if (j == 0) {
        frac = static_cast<float>(left_i + 1) - left_f;
      }
      /* right boundary */
      if (j == (right_i - left_i) - 1) {
        frac = right_f - static_cast<float>(left_i + j);
      }
      /* if the left and right boundaries are in the same cell */
      if ((right_i - left_i) == 1) {
        frac = delta;
      }
      value += input[IND(left_i + j, globalID.y)] * frac;
    }
    output[IND(globalID.x, globalID.y)] = value * normalization;
  }
}

extern "C" __global__ void resample_y(
  const float* input,
        float* output,
        size_t out_width,
        size_t out_height,
        size_t in_height)
{
  dim3 globalID(blockDim.x * blockIdx.x + threadIdx.x,
                blockDim.y * blockIdx.y + threadIdx.y);

  if (globalID.x < out_width &&  globalID.y < out_height) {
    float delta = in_height / static_cast<float>(out_height);
    float normalization = out_height / static_cast<float>(in_height);

    float left_f = globalID.y * delta;
    float right_f = (globalID.y + 1) * delta;

    int left_i = static_cast<int>(floor(left_f));
    int right_i = min(in_height, static_cast<size_t>(ceil(right_f)));

    float value = 0.f;
    
    for (int j = 0; j < (right_i - left_i); j++) {
      float frac = 1.f;

      /* left boundary */
      if (j == 0) {
        frac = static_cast<float>(left_i + 1) - left_f;
      }
      /* right boundary */
      if (j == (right_i - left_i) - 1) {
        frac = right_f - static_cast<float>(left_i + j);
      }
      /* if the left and right boundaries are in the same cell */
      if ((right_i - left_i) == 1) {
        frac = delta;
      }
      value += input[IND(globalID.x, left_i + j)] * frac;
    }
    output[IND(globalID.x, globalID.y)] = value * normalization;
  }
}

//extern "C" __global__ void resample_z(
//  const float* input,
//        float* output,
//        size_t out_width,
//        size_t out_height,
//        size_t out_depth,
//        size_t in_depth)
//{
//  dim3 globalID(blockDim.x * blockIdx.x + threadIdx.x,
//                blockDim.y * blockIdx.y + threadIdx.y,
//                blockDim.z * blockIdx.z + threadIdx.z);
//
//  if (globalID.x < out_width &&  globalID.y < out_height && globalID.z < out_depth) {
//    float delta = in_depth / static_cast<float>(out_depth);
//    float normalization = out_depth / static_cast<float>(in_depth);
//
//    float left_f = globalID.z * delta;
//    float right_f = (globalID.z + 1) * delta;
//
//    int left_i = static_cast<int>(floor(left_f));
//    int right_i = min(in_depth, static_cast<size_t>(ceil(right_f)));
//
//    float value = 0.f;
//    
//    for (int j = 0; j < (right_i - left_i); j++) {
//      float frac = 1.f;
//
//      /* left boundary */
//      if (j == 0) {
//        frac = static_cast<float>(left_i + 1) - left_f;
//      }
//      /* right boundary */
//      if (j == (right_i - left_i) - 1) {
//        frac = right_f - static_cast<float>(left_i + j);
//      }
//      /* if the left and right boundaries are in the same cell */
//      if ((right_i - left_i) == 1) {
//        frac = delta;
//      }
//      value += input[IND(globalID.x, globalID.y, left_i + j)] * frac;
//    }
//    output[IND(globalID.x, globalID.y, globalID.z)] = value * normalization;
//  }
//}

//extern "C" __global__ void resample_x_debug(
//  const float* input,
//  float* output,
//  size_t width,
//  size_t height,
//  size_t depth,
//  size_t resample_width)
//{
//  dim3 globalID(blockDim.x * blockIdx.x + threadIdx.x,
//    blockDim.y * blockIdx.y + threadIdx.y,
//    blockDim.z * blockIdx.z + threadIdx.z);
//
//  if (globalID.y < height && globalID.z < depth) {
//    for (size_t x = 0; x < width; x += 4) {
//      *((float4*)(&(output[IND(x, globalID.y, globalID.z)]))) =
//        *((const float4*)(&(input[IND(x, globalID.y, globalID.z)])));
//    }
//    for (size_t x = 0; x < width; x++) {
//      output[IND(x, globalID.y, globalID.z)] = input[IND(x, globalID.y, globalID.z)];
//    }
//  }
//}