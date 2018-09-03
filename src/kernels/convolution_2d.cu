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

/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <assert.h>
//#include <helper_cuda.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <cstdio>

#define __CUDACC__

#include <device_functions.h>
#include <math_functions.h>

#include "src/data_types/data_structs.h"

using namespace std;


#define MAX_KERNEL_LENGTH 51

__constant__ DataSize3 container_size;



////////////////////////////////////////////////////////////////////////////////
// Convolution kernel storage
////////////////////////////////////////////////////////////////////////////////
__constant__ float c_Kernel[MAX_KERNEL_LENGTH];

//extern "C" void setConvolutionKernel(float *h_Kernel, unsigned int kernel_length)
//{
//    cudaMemcpyToSymbol(c_Kernel, h_Kernel, kernel_length * sizeof(float));
//}


////////////////////////////////////////////////////////////////////////////////
// Row convolution filter
////////////////////////////////////////////////////////////////////////////////
#define   ROWS_BLOCKDIM_X 16
#define   ROWS_BLOCKDIM_Y 4
#define ROWS_RESULT_STEPS 4
#define   ROWS_HALO_STEPS 1

extern "C" __global__ void convolutionRowsKernel(
    float *d_Dst,
    const float *d_Src,
    int imageW,
    int imageH,
    int pitch,
    int kernel_radius
)
{

 /*   dim3 global_id(blockDim.x * blockIdx.x * ROWS_RESULT_STEPS + threadIdx.x,
        blockDim.y * blockIdx.y  + threadIdx.y);

    if (global_id.x > imageW)
        return;*/

    // Handle to thread block group
    __shared__ float s_Data[ROWS_BLOCKDIM_Y][(ROWS_RESULT_STEPS + 2 * ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X];

    //Offset to the left halo edge
    const int baseX = (blockIdx.x * ROWS_RESULT_STEPS - ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X + threadIdx.x;
    const int baseY = blockIdx.y * ROWS_BLOCKDIM_Y + threadIdx.y;

    d_Src += baseY * pitch + baseX;
    d_Dst += baseY * pitch + baseX;

    //Load main data
#pragma unroll

    for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++)
    {
        // Original NVIDIA version
        //s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = d_Src[i * ROWS_BLOCKDIM_X];

        // Modified version to allow for arbitrary image width
        size_t global_x = blockIdx.x * ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X + ROWS_BLOCKDIM_X*(i-ROWS_HALO_STEPS) + threadIdx.x;
        s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (global_x < imageW) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;
    }

    //Load left halo
#pragma unroll

    for (int i = 0; i < ROWS_HALO_STEPS; i++)
    {
        s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (baseX >= -i * ROWS_BLOCKDIM_X) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;
    }

    //Load right halo
#pragma unroll

    for (int i = ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS + ROWS_HALO_STEPS; i++)
    {
        s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (imageW - baseX > i * ROWS_BLOCKDIM_X) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;

       
       // Testing 
      /*  size_t global_x = blockIdx.x * ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X + ROWS_BLOCKDIM_X*(i-5) + threadIdx.x;

        s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (global_x < imageW) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;

        if (global_x > imageW)
            std::printf("Bx:%u tx:%u iter:%d global.x: %lu data:%f \n", blockIdx.x, threadIdx.x, i, static_cast<unsigned long>(global_x), s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X]);
*/
    }

    //Compute and store results


    // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< //
    // -------------------------------------------------------- //
    __syncthreads();
    // -------------------------------------------------------- //
    // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< //

#pragma unroll

    for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++)
    {
        size_t global_x = blockIdx.x * ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X + ROWS_BLOCKDIM_X*(i-ROWS_HALO_STEPS) + threadIdx.x;

        if (global_x >= imageW)
            return;

        float sum = 0;

#pragma unroll

        for (int j = -kernel_radius; j <= kernel_radius; j++)
        {
            sum += c_Kernel[kernel_radius - j] * s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X + j];
        }

        d_Dst[i * ROWS_BLOCKDIM_X] = sum;
    }
}




////////////////////////////////////////////////////////////////////////////////
// Column convolution filter
////////////////////////////////////////////////////////////////////////////////
#define   COLUMNS_BLOCKDIM_X 4
#define   COLUMNS_BLOCKDIM_Y 16
#define COLUMNS_RESULT_STEPS 4
#define   COLUMNS_HALO_STEPS 1

extern "C" __global__ void convolutionColumnsKernel(
    float *d_Dst,
    const float *d_Src,
    int imageW,
    int imageH,
    int pitch,
    int kernel_radius
)
{
    __shared__ float s_Data[COLUMNS_BLOCKDIM_X][(COLUMNS_RESULT_STEPS + 2 * COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + 0];

    //Offset to the upper halo edge
    const int baseX = blockIdx.x * COLUMNS_BLOCKDIM_X + threadIdx.x;
    const int baseY = (blockIdx.y * COLUMNS_RESULT_STEPS - COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + threadIdx.y;

    d_Src += baseY * pitch + baseX;
    d_Dst += baseY * pitch + baseX;

    //Main data
#pragma unroll

    for (int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++)
    {
        // Original NVIDIA version
        //s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = d_Src[i * COLUMNS_BLOCKDIM_Y * pitch];

        // Modified version to allow for arbitrary image height
        size_t global_y = blockIdx.y * COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y + COLUMNS_BLOCKDIM_Y*(i-COLUMNS_HALO_STEPS) + threadIdx.y;
        s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y]= (global_y < imageH) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitch] : 0;

    }

    //Upper halo
#pragma unroll

    for (int i = 0; i < COLUMNS_HALO_STEPS; i++)
    {
        s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = (baseY >= -i * COLUMNS_BLOCKDIM_Y) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitch] : 0;
    }

    //Lower halo
#pragma unroll

    for (int i = COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS + COLUMNS_HALO_STEPS; i++)
    {
        s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y]= (imageH - baseY > i * COLUMNS_BLOCKDIM_Y) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitch] : 0;


    }

    // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< //
    // -------------------------------------------------------- //
    __syncthreads();
    // -------------------------------------------------------- //
    // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< //

#pragma unroll

    for (int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++)
    {

        size_t global_y =  blockIdx.y * COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y + COLUMNS_BLOCKDIM_Y*(i-COLUMNS_HALO_STEPS) + threadIdx.y;

        //std::printf("By:%u ty:%u iter:%d global.y: %lu\n", blockIdx.y, threadIdx.y, i, static_cast<unsigned long>(global_y));

        if (global_y >= imageH)
            return;

        float sum = 0;
#pragma unroll

        for (int j = -kernel_radius; j <= kernel_radius; j++)
        {
            sum += c_Kernel[kernel_radius - j] * s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y + j];
        }

        d_Dst[i * COLUMNS_BLOCKDIM_Y * pitch] = sum;

        //std::printf("out: %d \n", i * COLUMNS_BLOCKDIM_Y * pitch);
    }
}


