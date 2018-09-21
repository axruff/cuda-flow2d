/**
* @file    2D Autocorrelation flow using NVIDIA CUDA
* @author  Institute for Photon Science and Synchrotron Radiation, Karlsruhe Institute of Technology
*
* @date    2018
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

#include <iostream>

#include <device_functions.h>
#include <math_functions.h>


#include "src/data_types/data_structs.h"



#define IND(X, Y) ((Y) * (container_size.pitch / sizeof(float)) + (X))
#define EIND(X, Y) ((Y) * (extended_pitch) + (X))
#define SIND(X, Y) ((((Y) + radius_2)) * shared_block_size.x + ((X) + radius_2))



__constant__ DataSize3 container_size;


extern __shared__ float shared[];



extern "C" __global__ void correlation_2d(
        const float* input,
        const size_t width,
        const size_t height,
        const size_t window_size,
        const size_t extended_pitch,
        float* corr_ext
        )
{
    dim3 global_id(blockDim.x * blockIdx.x + threadIdx.x,
        blockDim.y * blockIdx.y + threadIdx.y);

    const int radius_2 = window_size / 2.0;

    dim3 shared_block_size(
        blockDim.x + 2 * radius_2,
        blockDim.y + 2 * radius_2
        );

    /* Load data to the shared memoty */
    size_t global_x = global_id.x < width ? global_id.x : 2 * width - global_id.x - 2;
    size_t global_y = global_id.y < height ? global_id.y : 2 * height - global_id.y - 2;


    /* Main area */
    shared[SIND(threadIdx.x, threadIdx.y)] = input[IND(global_x, global_y)];

    /* Left slice */
    if (threadIdx.x < radius_2) {
        int offset = blockDim.x * blockIdx.x - radius_2 + threadIdx.x;
        size_t global_x_l = offset >= 0 ? offset : -offset;
        shared[SIND(-radius_2 + threadIdx.x, threadIdx.y)] = input[IND(global_x_l, global_y)];
    }

    /* Right slice */
    if (threadIdx.x > blockDim.x - 1 - radius_2) {
        int index = blockDim.x - threadIdx.x;
        int offset = blockDim.x *(blockIdx.x + 1) + radius_2 - index;
        size_t global_x_r = offset < width ? offset : 2 * width - offset - 2;
        shared[SIND(radius_2 + threadIdx.x, threadIdx.y)] = input[IND(global_x_r, global_y)];
    }

    /* Upper slice */
    if (threadIdx.y < radius_2) {
        int offset = blockDim.y * blockIdx.y - radius_2 + threadIdx.y;
        size_t global_y_u = offset >= 0 ? offset : -offset;
        shared[SIND(threadIdx.x, -radius_2 + threadIdx.y)] = input[IND(global_x, global_y_u)];
    }

    /* Bottom slice */
    if (threadIdx.y > blockDim.y - 1 - radius_2) {
        int index = blockDim.y - threadIdx.y;
        int offset = blockDim.y *(blockIdx.y + 1) + radius_2 - index;
        size_t global_y_b = offset < height ? offset : 2 * height - offset - 2;
        shared[SIND(threadIdx.x, radius_2 + threadIdx.y)] = input[IND(global_x, global_y_b)];
    }

    /* 4 corners */
    {
        int global_x_c;
        int global_y_c;

        if (threadIdx.x < radius_2 && threadIdx.y < radius_2) {

            global_x_c = blockDim.x * blockIdx.x - radius_2 + threadIdx.x;
            global_x_c = global_x_c > 0 ? global_x_c : -global_x_c;

            global_y_c = blockDim.y * blockIdx.y - radius_2 + threadIdx.y;
            global_y_c = global_y_c > 0 ? global_y_c : -global_y_c;

            /* Front upper left */
            shared[SIND(threadIdx.x - radius_2, threadIdx.y - radius_2)] =
                input[IND(global_x_c, global_y_c)];

            /* Front upper right */
            global_x_c = blockDim.x *(blockIdx.x + 1) + threadIdx.x;
            global_x_c = global_x_c < width ? global_x_c : 2 * width - global_x_c - 2;
            shared[SIND(blockDim.x + threadIdx.x, threadIdx.y - radius_2)] =
                input[IND(global_x_c, global_y_c)];

            /* Front bottom right */
            global_y_c = blockDim.y *(blockIdx.y + 1) + threadIdx.y;
            global_y_c = global_y_c < height ? global_y_c : 2 * height - global_y_c - 2;
            shared[SIND(blockDim.x + threadIdx.x, blockDim.y + threadIdx.y)] =
                input[IND(global_x_c, global_y_c)];

            /* Front bottom left */
            global_x_c = blockDim.x * blockIdx.x - radius_2 + threadIdx.x;
            global_x_c = global_x_c > 0 ? global_x_c : -global_x_c;
            shared[SIND(threadIdx.x - radius_2, blockDim.y + threadIdx.y)] =
                input[IND(global_x_c, global_y_c)];

        }
    }




    // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< //
    // -------------------------------------------------------- //
    __syncthreads();
    // -------------------------------------------------------- //
    // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< //



    if (global_id.x < width && global_id.y < height) {

        float norm_sum = 0.0f;
        float val = 0.0f;

        // Normalization coefficient
        for (int i = -radius_2; i < radius_2; i++) {
            for (int j = -radius_2; j < radius_2; j++) {
                val = shared[SIND(threadIdx.x + i, threadIdx.y + j)];
                norm_sum += val*val;
            }
        }


        int shift_x = threadIdx.x - radius_2;
        int shift_y = threadIdx.y - radius_2;

        for (int i = -radius_2+1; i < radius_2+1; i++) {
            for (int j = -radius_2+1; j < radius_2+1; j++) {

                float sum_c = 0.0f;

                for (int k = 0; k < window_size; k++) {
                    for (int t = 0; t < window_size; t++) {

                        if (((k-i+1) < 0 || (k-i+1)> window_size-1 || (t-j+1) < 0 || (t-j+1)>window_size-1))
                            continue;

                        sum_c += shared[SIND(t+shift_x, k+shift_y)] * shared[SIND(t-j+1+shift_x, k-i+1+shift_y)];


                        //sum_c += 1.0;


                    }
                }

                corr_ext[EIND(global_id.x*window_size+j+radius_2-1, global_id.y*window_size+i+radius_2-1)] = sum_c / norm_sum;

            }

        }
        


    }
        
    

}


extern "C" __global__ void find_peak_2d(
    const float* input,
    const size_t width,
    const size_t height,
    const size_t window_size,
    const size_t min_distance,
    const size_t extended_pitch,
    float* corr_ext
    )
{
    dim3 global_id(blockDim.x * blockIdx.x + threadIdx.x,
        blockDim.y * blockIdx.y + threadIdx.y);

    const int radius_2 = min_distance;


    dim3 shared_block_size(
        blockDim.x + 2 * min_distance,
        blockDim.y + 2 * min_distance
        );

    /* Load data to the shared memoty */
    size_t global_x = global_id.x < width ? global_id.x : 2 * width - global_id.x - 2;
    size_t global_y = global_id.y < height ? global_id.y : 2 * height - global_id.y - 2;


    

    /* Main area */
    shared[SIND(threadIdx.x, threadIdx.y)] = input[EIND(global_x, global_y)];

    /* Left slice */
    if (threadIdx.x < radius_2) {
        int offset = blockDim.x * blockIdx.x - radius_2 + threadIdx.x;
        size_t global_x_l = offset >= 0 ? offset : -offset;
        shared[SIND(-radius_2 + threadIdx.x, threadIdx.y)] = input[EIND(global_x_l, global_y)];
    }

    /* Right slice */
    if (threadIdx.x > blockDim.x - 1 - radius_2) {
        int index = blockDim.x - threadIdx.x;
        int offset = blockDim.x *(blockIdx.x + 1) + radius_2 - index;
        size_t global_x_r = offset < width ? offset : 2 * width - offset - 2;
        shared[SIND(radius_2 + threadIdx.x, threadIdx.y)] = input[EIND(global_x_r, global_y)];
    }

    /* Upper slice */
    if (threadIdx.y < radius_2) {
        int offset = blockDim.y * blockIdx.y - radius_2 + threadIdx.y;
        size_t global_y_u = offset >= 0 ? offset : -offset;
        shared[SIND(threadIdx.x, -radius_2 + threadIdx.y)] = input[EIND(global_x, global_y_u)];
    }

    /* Bottom slice */
    if (threadIdx.y > blockDim.y - 1 - radius_2) {
        int index = blockDim.y - threadIdx.y;
        int offset = blockDim.y *(blockIdx.y + 1) + radius_2 - index;
        size_t global_y_b = offset < height ? offset : 2 * height - offset - 2;
        shared[SIND(threadIdx.x, radius_2 + threadIdx.y)] = input[EIND(global_x, global_y_b)];
    }

    /* 4 corners */
    {
        int global_x_c;
        int global_y_c;

        if (threadIdx.x < radius_2 && threadIdx.y < radius_2) {

            global_x_c = blockDim.x * blockIdx.x - radius_2 + threadIdx.x;
            global_x_c = global_x_c > 0 ? global_x_c : -global_x_c;

            global_y_c = blockDim.y * blockIdx.y - radius_2 + threadIdx.y;
            global_y_c = global_y_c > 0 ? global_y_c : -global_y_c;

            /* Front upper left */
            shared[SIND(threadIdx.x - radius_2, threadIdx.y - radius_2)] =
                input[EIND(global_x_c, global_y_c)];

            /* Front upper right */
            global_x_c = blockDim.x *(blockIdx.x + 1) + threadIdx.x;
            global_x_c = global_x_c < width ? global_x_c : 2 * width - global_x_c - 2;
            shared[SIND(blockDim.x + threadIdx.x, threadIdx.y - radius_2)] =
                input[EIND(global_x_c, global_y_c)];

            /* Front bottom right */
            global_y_c = blockDim.y *(blockIdx.y + 1) + threadIdx.y;
            global_y_c = global_y_c < height ? global_y_c : 2 * height - global_y_c - 2;
            shared[SIND(blockDim.x + threadIdx.x, blockDim.y + threadIdx.y)] =
                input[EIND(global_x_c, global_y_c)];

            /* Front bottom left */
            global_x_c = blockDim.x * blockIdx.x - radius_2 + threadIdx.x;
            global_x_c = global_x_c > 0 ? global_x_c : -global_x_c;
            shared[SIND(threadIdx.x - radius_2, blockDim.y + threadIdx.y)] =
                input[EIND(global_x_c, global_y_c)];

        }
    }

  

    // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< //
    // -------------------------------------------------------- //
    __syncthreads();
    // -------------------------------------------------------- //
    // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< //

    float eps = 1e-4;

    if (global_id.x < width && global_id.y < height) {

        float local_max = 0.0f;


        for (int i=(-min_distance); i <= static_cast<int>(min_distance); i++) {
            for (int j=(-min_distance); j <= static_cast<int>(min_distance); j++) {

                int lx = threadIdx.x + i;
                int ly = threadIdx.y + j;


                float val = shared[SIND(lx, ly)];

                //if (blockIdx.x == 1 && blockIdx.y == 0 && lx == 9 && ly == 9)
                //   std::printf("lx:%u ly:%u dval: %f \n", lx, ly, val);

                if (val > local_max)
                    local_max = val;


                
            }
        }

        //corr_ext[EIND(global_id.x, global_id.y)] = global_id.x;
        //corr_ext[EIND(global_id.x, global_id.y)] = local_max;
        //corr_ext[EIND(global_id.x, global_id.y)] = shared[SIND(threadIdx.x, threadIdx.y)];

        float val = shared[SIND(threadIdx.x, threadIdx.y)];
        if (fabsf(val - local_max) < eps)
            corr_ext[EIND(global_id.x, global_id.y)] = val;
        else 
            corr_ext[EIND(global_id.x, global_id.y)] = 0.0f;


        

    }



}


extern "C" __global__ void select_peak_2d(
    const float* input,
    const size_t width,
    const size_t height,
    const size_t window_size,
    const size_t extended_pitch,
    float* flow_x,
    float* flow_y,
    float* corr
    )
{
    dim3 global_id(blockDim.x * blockIdx.x + threadIdx.x,
        blockDim.y * blockIdx.y + threadIdx.y);


    size_t global_x = global_id.x < width ? global_id.x : 2 * width - global_id.x - 2;
    size_t global_y = global_id.y < height ? global_id.y : 2 * height - global_id.y - 2;


    float m1 = 0.0f;
    float m2 = 0.0f;

    float x1 = 0.0f;
    float x2 = 0.0f;

    float y1 = 0.0f;
    float y2 = 0.0f;

    const float EPS = 1e-5;


    if (global_id.x < width && global_id.y < height) {

        for (int i=0; i<window_size; i++) {
            for (int j=0; j<window_size; j++) {
          

                float val = input[EIND(global_x *window_size+i, global_y*window_size+j)];


                if (val > m1 && (fabsf(val - 1.0f) > EPS)) {
                    m1 = val;
                    x1 = i;
                    y1 = j;
                }

      
                if (val > m2 && (fabsf(val - 1.0f) > EPS) && ((i != x1) || (j != y1))) {
                    m2 = val;
                    x2 = i;
                    y2 = j;
                }

          
            }
        }

 
        float m = 0.0f;
        float x = 0.0f;
        float y = 0.0f;

        m = (m1 > m2) ? m1 : m2;
        x = (m1 > m2) ? x1 : x2;
        y = (m1 > m2) ? y1 : y2;


        corr[IND(global_id.x, global_id.y )] = m;
        flow_x[IND(global_id.x, global_id.y)] = x - window_size / 2.0;
        flow_y[IND(global_id.x, global_id.y)] = y - window_size / 2.0;
    }

}