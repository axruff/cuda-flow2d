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
        float* flow_x,
        float* flow_y,
        float* corr,
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
        flow_x[IND(global_id.x, global_id.y)] = blockIdx.x;
        flow_y[IND(global_id.x, global_id.y)] = blockIdx.y;
        corr[IND(global_id.x, global_id.y)] = blockIdx.x*blockIdx.y;

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