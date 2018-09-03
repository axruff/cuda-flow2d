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

#include <device_functions.h>
#include <math_functions.h>

#include "src/data_types/data_structs.h"

//#define IND(X, Y, Z) (((Z) * container_size.height + (Y)) * (container_size.pitch / sizeof(float)) + (X))
//#define SIND(X, Y, Z) ((((Z) + 1) * shared_block_size.y + ((Y) + 1)) * shared_block_size.x + ((X) + 1))

#define IND(X, Y) ((Y) * (container_size.pitch / sizeof(float)) + (X)) 
#define SIND(X, Y) ((((Y) + 1)) * shared_block_size.x + ((X) + 1))

__constant__ DataSize3 container_size;

extern __shared__ float shared[];

extern "C" __global__ void compute_phi_ksi(
  const float* frame_0,
  const float* frame_1,
  const float* flow_u,
  const float* flow_v,
  const float* flow_du,
  const float* flow_dv,
        size_t width,
        size_t height,
        float  hx,
        float  hy,
        float  equation_smootness,
        float  equation_data,
        float* phi,
        float* ksi)
{
  dim3 shared_block_size(
    blockDim.x + 2,
    blockDim.y + 2);

  float* shared_frame_0 = &shared[0 * shared_block_size.x * shared_block_size.y];
  float* shared_frame_1 = &shared[1 * shared_block_size.x * shared_block_size.y];
  float* shared_flow_u  = &shared[2 * shared_block_size.x * shared_block_size.y];
  float* shared_flow_v  = &shared[3 * shared_block_size.x * shared_block_size.y];
  float* shared_flow_du = &shared[4 * shared_block_size.x * shared_block_size.y];
  float* shared_flow_dv = &shared[5 * shared_block_size.x * shared_block_size.y];

  dim3 global_id(
    blockDim.x * blockIdx.x + threadIdx.x,
    blockDim.y * blockIdx.y + threadIdx.y);

  /* Load the main area of datasets */
  size_t global_x = global_id.x < width ? global_id.x : 2 * width - global_id.x - 2;
  size_t global_y = global_id.y < height ? global_id.y : 2 * height - global_id.y - 2;
  {
    shared_frame_0[SIND(threadIdx.x, threadIdx.y)] = frame_0[IND(global_x, global_y)];
    shared_frame_1[SIND(threadIdx.x, threadIdx.y)] = frame_1[IND(global_x, global_y)];
    shared_flow_u [SIND(threadIdx.x, threadIdx.y)] = flow_u [IND(global_x, global_y)];
    shared_flow_v [SIND(threadIdx.x, threadIdx.y)] = flow_v [IND(global_x, global_y)];
    shared_flow_du[SIND(threadIdx.x, threadIdx.y)] = flow_du[IND(global_x, global_y)];
    shared_flow_dv[SIND(threadIdx.x, threadIdx.y)] = flow_dv[IND(global_x, global_y)];
  }

  /* Load the left slice */
  if (threadIdx.x == 0) {
    int offset = global_x - 1;
    size_t global_x_l = offset >= 0 ? offset : -offset;
    shared_frame_0[SIND(threadIdx.x - 1, threadIdx.y)] = frame_0[IND(global_x_l, global_y)];
    shared_frame_1[SIND(threadIdx.x - 1, threadIdx.y)] = frame_1[IND(global_x_l, global_y)];
    shared_flow_u [SIND(threadIdx.x - 1, threadIdx.y)] = flow_u [IND(global_x_l, global_y)];
    shared_flow_v [SIND(threadIdx.x - 1, threadIdx.y)] = flow_v [IND(global_x_l, global_y)];
    shared_flow_du[SIND(threadIdx.x - 1, threadIdx.y)] = flow_du[IND(global_x_l, global_y)];
    shared_flow_dv[SIND(threadIdx.x - 1, threadIdx.y)] = flow_dv[IND(global_x_l, global_y)];

  }

  /* Load the right slice */
  if (threadIdx.x == blockDim.x - 1) {
    int offset = global_x + 1;
    size_t global_x_r = offset < width ? offset : 2 * width - offset - 2;
    shared_frame_0[SIND(threadIdx.x + 1, threadIdx.y)] = frame_0[IND(global_x_r, global_y)];
    shared_frame_1[SIND(threadIdx.x + 1, threadIdx.y)] = frame_1[IND(global_x_r, global_y)];
    shared_flow_u [SIND(threadIdx.x + 1, threadIdx.y)] = flow_u [IND(global_x_r, global_y)];
    shared_flow_v [SIND(threadIdx.x + 1, threadIdx.y)] = flow_v [IND(global_x_r, global_y)];
    shared_flow_du[SIND(threadIdx.x + 1, threadIdx.y)] = flow_du[IND(global_x_r, global_y)];
    shared_flow_dv[SIND(threadIdx.x + 1, threadIdx.y)] = flow_dv[IND(global_x_r, global_y)];
  }

  /* Load the upper slice */
  if (threadIdx.y == 0) {
    int offset = global_y - 1;
    size_t global_y_u = offset > 0 ? offset : -offset;
    shared_frame_0[SIND(threadIdx.x, threadIdx.y - 1)] = frame_0[IND(global_x, global_y_u)];
    shared_frame_1[SIND(threadIdx.x, threadIdx.y - 1)] = frame_1[IND(global_x, global_y_u)];
    shared_flow_u [SIND(threadIdx.x, threadIdx.y - 1)] = flow_u [IND(global_x, global_y_u)];
    shared_flow_v [SIND(threadIdx.x, threadIdx.y - 1)] = flow_v [IND(global_x, global_y_u)];
    shared_flow_du[SIND(threadIdx.x, threadIdx.y - 1)] = flow_du[IND(global_x, global_y_u)];
    shared_flow_dv[SIND(threadIdx.x, threadIdx.y - 1)] = flow_dv[IND(global_x, global_y_u)];
  }

  /* Load the bottom slice */
  if (threadIdx.y == blockDim.y - 1) {
    int offset = global_y + 1;
    size_t global_y_b = offset < height ? offset : 2 * height - offset - 2;
    shared_frame_0[SIND(threadIdx.x, threadIdx.y + 1)] = frame_0[IND(global_x, global_y_b)];
    shared_frame_1[SIND(threadIdx.x, threadIdx.y + 1)] = frame_1[IND(global_x, global_y_b)];
    shared_flow_u [SIND(threadIdx.x, threadIdx.y + 1)] = flow_u [IND(global_x, global_y_b)];
    shared_flow_v [SIND(threadIdx.x, threadIdx.y + 1)] = flow_v [IND(global_x, global_y_b)];
    shared_flow_du[SIND(threadIdx.x, threadIdx.y + 1)] = flow_du[IND(global_x, global_y_b)];
    shared_flow_dv[SIND(threadIdx.x, threadIdx.y + 1)] = flow_dv[IND(global_x, global_y_b)];
  }


  __syncthreads();

  /* Compute flow-driven terms */
  if (global_id.x < width && global_id.y < height) {

    float dux =
      (shared_flow_u [SIND(threadIdx.x + 1, threadIdx.y)] - shared_flow_u [SIND(threadIdx.x - 1, threadIdx.y)] +
       shared_flow_du[SIND(threadIdx.x + 1, threadIdx.y)] - shared_flow_du[SIND(threadIdx.x - 1, threadIdx.y)]) /
      (2.f * hx);
    float duy =
      (shared_flow_u [SIND(threadIdx.x, threadIdx.y + 1)] - shared_flow_u [SIND(threadIdx.x, threadIdx.y - 1)] +
       shared_flow_du[SIND(threadIdx.x, threadIdx.y + 1)] - shared_flow_du[SIND(threadIdx.x, threadIdx.y - 1)]) /
      (2.f * hy);

    float dvx =
      (shared_flow_v [SIND(threadIdx.x + 1, threadIdx.y)] - shared_flow_v [SIND(threadIdx.x - 1, threadIdx.y)] +
       shared_flow_dv[SIND(threadIdx.x + 1, threadIdx.y)] - shared_flow_dv[SIND(threadIdx.x - 1, threadIdx.y)]) /
      (2.f * hx);
    float dvy =
      (shared_flow_v [SIND(threadIdx.x, threadIdx.y + 1)] - shared_flow_v [SIND(threadIdx.x, threadIdx.y - 1)] +
       shared_flow_dv[SIND(threadIdx.x, threadIdx.y + 1)] - shared_flow_dv[SIND(threadIdx.x, threadIdx.y - 1)]) /
      (2.f * hy);


    /* Flow-driven term phi */
    phi[IND(global_id.x, global_id.y)] =
      1.f / (2.f * sqrtf(dux*dux + duy*duy  + dvx*dvx + dvy*dvy + equation_smootness * equation_smootness));

    float fx = 
      (shared_frame_0[SIND(threadIdx.x + 1, threadIdx.y)] - shared_frame_0[SIND(threadIdx.x - 1, threadIdx.y)] +
       shared_frame_1[SIND(threadIdx.x + 1, threadIdx.y)] - shared_frame_1[SIND(threadIdx.x - 1, threadIdx.y)]) / 
       (4.f * hx);
    float fy = 
      (shared_frame_0[SIND(threadIdx.x, threadIdx.y + 1)] - shared_frame_0[SIND(threadIdx.x, threadIdx.y - 1)] +
       shared_frame_1[SIND(threadIdx.x, threadIdx.y + 1)] - shared_frame_1[SIND(threadIdx.x, threadIdx.y - 1)]) / 
       (4.f * hy);

    float ft = 
      shared_frame_1[SIND(threadIdx.x, threadIdx.y)] - shared_frame_0[SIND(threadIdx.x, threadIdx.y)];

    float J11 = fx * fx;
    float J22 = fy * fy;
    float J33 = ft * ft;
    float J12 = fx * fy;
    float J13 = fx * ft;
    float J23 = fy * ft;

    float& du = shared_flow_du[SIND(threadIdx.x, threadIdx.y)];
    float& dv = shared_flow_dv[SIND(threadIdx.x, threadIdx.y)];


    float s =
      (J11 * du + J12 * dv + J13) * du +
      (J12 * du + J22 * dv + J23) * dv +
      (J13 * du + J23 * dv + J33);

    s = (s > 0) * s;

    /* Penalizer function for the data term ksi */
    ksi[IND(global_id.x, global_id.y)] =
      1.f / (2.f * sqrtf(s + equation_data * equation_data));
  }
}

extern "C" __global__ void solve_2d(
  const float* frame_0,
  const float* frame_1,
  const float* flow_u,
  const float* flow_v,
  const float* flow_du,
  const float* flow_dv,
  const float* phi,
  const float* ksi,
        size_t width,
        size_t height,
        float  hx,
        float  hy,
        float  equation_alpha,
        float* temp_du,
        float* temp_dv)
{
  dim3 shared_block_size(
    blockDim.x + 2,
    blockDim.y + 2);

  float* shared_frame_0 = &shared[0 * shared_block_size.x * shared_block_size.y];
  float* shared_frame_1 = &shared[1 * shared_block_size.x * shared_block_size.y];
  float* shared_flow_u  = &shared[2 * shared_block_size.x * shared_block_size.y];
  float* shared_flow_v  = &shared[3 * shared_block_size.x * shared_block_size.y];
  float* shared_flow_du = &shared[4 * shared_block_size.x * shared_block_size.y];
  float* shared_flow_dv = &shared[5 * shared_block_size.x * shared_block_size.y];
  float* shared_phi     = &shared[6 * shared_block_size.x * shared_block_size.y];
  float* shared_ksi     = &shared[7 * shared_block_size.x * shared_block_size.y];


  dim3 global_id(
    blockDim.x * blockIdx.x + threadIdx.x,
    blockDim.y * blockIdx.y + threadIdx.y);

  /* Load the main area of datasets */
  size_t global_x = global_id.x < width ? global_id.x : 2 * width - global_id.x - 2;
  size_t global_y = global_id.y < height ? global_id.y : 2 * height - global_id.y - 2;
  {
    shared_frame_0[SIND(threadIdx.x, threadIdx.y)] = frame_0[IND(global_x, global_y)];
    shared_frame_1[SIND(threadIdx.x, threadIdx.y)] = frame_1[IND(global_x, global_y)];
    shared_flow_u [SIND(threadIdx.x, threadIdx.y)] = flow_u [IND(global_x, global_y)];
    shared_flow_v [SIND(threadIdx.x, threadIdx.y)] = flow_v [IND(global_x, global_y)];
    shared_flow_du[SIND(threadIdx.x, threadIdx.y)] = flow_du[IND(global_x, global_y)];
    shared_flow_dv[SIND(threadIdx.x, threadIdx.y)] = flow_dv[IND(global_x, global_y)];
    shared_phi    [SIND(threadIdx.x, threadIdx.y)] =     phi[IND(global_x, global_y)];
    shared_ksi    [SIND(threadIdx.x, threadIdx.y)] =     ksi[IND(global_x, global_y)];
  }

  /* Load the left slice */
  if (threadIdx.x == 0) {
    int offset = global_x - 1;
    size_t global_x_l = offset >= 0 ? offset : -offset;
    shared_frame_0[SIND(threadIdx.x - 1, threadIdx.y)] = frame_0[IND(global_x_l, global_y)];
    shared_frame_1[SIND(threadIdx.x - 1, threadIdx.y)] = frame_1[IND(global_x_l, global_y)];
    shared_flow_u [SIND(threadIdx.x - 1, threadIdx.y)] = flow_u [IND(global_x_l, global_y)];
    shared_flow_v [SIND(threadIdx.x - 1, threadIdx.y)] = flow_v [IND(global_x_l, global_y)];
    shared_flow_du[SIND(threadIdx.x - 1, threadIdx.y)] = flow_du[IND(global_x_l, global_y)];
    shared_flow_dv[SIND(threadIdx.x - 1, threadIdx.y)] = flow_dv[IND(global_x_l, global_y)];
    shared_phi    [SIND(threadIdx.x - 1, threadIdx.y)] =     phi[IND(global_x_l, global_y)];
    shared_ksi    [SIND(threadIdx.x - 1, threadIdx.y)] =     ksi[IND(global_x_l, global_y)];
  }

  /* Load the right slice */
  if (threadIdx.x == blockDim.x - 1) {
    int offset = global_x + 1;
    size_t global_x_r = offset < width ? offset : 2 * width - offset - 2;
    shared_frame_0[SIND(threadIdx.x + 1, threadIdx.y)] = frame_0[IND(global_x_r, global_y)];
    shared_frame_1[SIND(threadIdx.x + 1, threadIdx.y)] = frame_1[IND(global_x_r, global_y)];
    shared_flow_u [SIND(threadIdx.x + 1, threadIdx.y)] = flow_u [IND(global_x_r, global_y)];
    shared_flow_v [SIND(threadIdx.x + 1, threadIdx.y)] = flow_v [IND(global_x_r, global_y)];
    shared_flow_du[SIND(threadIdx.x + 1, threadIdx.y)] = flow_du[IND(global_x_r, global_y)];
    shared_flow_dv[SIND(threadIdx.x + 1, threadIdx.y)] = flow_dv[IND(global_x_r, global_y)];
    shared_phi    [SIND(threadIdx.x + 1, threadIdx.y)] =     phi[IND(global_x_r, global_y)];
    shared_ksi    [SIND(threadIdx.x + 1, threadIdx.y)] =     ksi[IND(global_x_r, global_y)];
  }

  /* Load the upper slice */
  if (threadIdx.y == 0) {
    int offset = global_y - 1;
    size_t global_y_u = offset > 0 ? offset : -offset;
    shared_frame_0[SIND(threadIdx.x, threadIdx.y - 1)] = frame_0[IND(global_x, global_y_u)];
    shared_frame_1[SIND(threadIdx.x, threadIdx.y - 1)] = frame_1[IND(global_x, global_y_u)];
    shared_flow_u [SIND(threadIdx.x, threadIdx.y - 1)] = flow_u [IND(global_x, global_y_u)];
    shared_flow_v [SIND(threadIdx.x, threadIdx.y - 1)] = flow_v [IND(global_x, global_y_u)];
    shared_flow_du[SIND(threadIdx.x, threadIdx.y - 1)] = flow_du[IND(global_x, global_y_u)];
    shared_flow_dv[SIND(threadIdx.x, threadIdx.y - 1)] = flow_dv[IND(global_x, global_y_u)];
    shared_phi    [SIND(threadIdx.x, threadIdx.y - 1)] =     phi[IND(global_x, global_y_u)];
    shared_ksi    [SIND(threadIdx.x, threadIdx.y - 1)] =     ksi[IND(global_x, global_y_u)];
  }

  /* Load the bottom slice */
  if (threadIdx.y == blockDim.y - 1) {
    int offset = global_y + 1;
    size_t global_y_b = offset < height ? offset : 2 * height - offset - 2;
    shared_frame_0[SIND(threadIdx.x, threadIdx.y + 1)] = frame_0[IND(global_x, global_y_b)];
    shared_frame_1[SIND(threadIdx.x, threadIdx.y + 1)] = frame_1[IND(global_x, global_y_b)];
    shared_flow_u [SIND(threadIdx.x, threadIdx.y + 1)] = flow_u [IND(global_x, global_y_b)];
    shared_flow_v [SIND(threadIdx.x, threadIdx.y + 1)] = flow_v [IND(global_x, global_y_b)];
    shared_flow_du[SIND(threadIdx.x, threadIdx.y + 1)] = flow_du[IND(global_x, global_y_b)];
    shared_flow_dv[SIND(threadIdx.x, threadIdx.y + 1)] = flow_dv[IND(global_x, global_y_b)];
    shared_phi    [SIND(threadIdx.x, threadIdx.y + 1)] =     phi[IND(global_x, global_y_b)];
    shared_ksi    [SIND(threadIdx.x, threadIdx.y + 1)] =     ksi[IND(global_x, global_y_b)];
  }


  __syncthreads();

  if (global_id.x < width && global_id.y < height) {
    /* Compute derivatives */

    float fx =
        (shared_frame_0[SIND(threadIdx.x + 1, threadIdx.y)] - shared_frame_0[SIND(threadIdx.x - 1, threadIdx.y)] +
        shared_frame_1[SIND(threadIdx.x + 1, threadIdx.y)] - shared_frame_1[SIND(threadIdx.x - 1, threadIdx.y)]) /
        (4.f * hx);
    float fy =
        (shared_frame_0[SIND(threadIdx.x, threadIdx.y + 1)] - shared_frame_0[SIND(threadIdx.x, threadIdx.y - 1)] +
        shared_frame_1[SIND(threadIdx.x, threadIdx.y + 1)] - shared_frame_1[SIND(threadIdx.x, threadIdx.y - 1)]) /
        (4.f * hy);

    float ft =
        shared_frame_1[SIND(threadIdx.x, threadIdx.y)] - shared_frame_0[SIND(threadIdx.x, threadIdx.y)];

    float J11 = fx * fx;
    float J22 = fy * fy;
    float J33 = ft * ft;
    float J12 = fx * fy;
    float J13 = fx * ft;
    float J23 = fy * ft;


    /* Compute weights */
    float hx_2 = equation_alpha / (hx * hx);
    float hy_2 = equation_alpha / (hy * hy);

    
    float xp = (global_id.x < width - 1)  * hx_2;
    float xm = (global_id.x > 0)          * hx_2;
    float yp = (global_id.y < height - 1) * hy_2;
    float ym = (global_id.y > 0)          * hy_2;


    float phi_xp = (shared_phi[SIND(threadIdx.x + 1, threadIdx.y)] + shared_phi[SIND(threadIdx.x, threadIdx.y)]) / 2.f;
    float phi_xm = (shared_phi[SIND(threadIdx.x - 1, threadIdx.y)] + shared_phi[SIND(threadIdx.x, threadIdx.y)]) / 2.f;
    float phi_yp = (shared_phi[SIND(threadIdx.x, threadIdx.y + 1)] + shared_phi[SIND(threadIdx.x, threadIdx.y)]) / 2.f;
    float phi_ym = (shared_phi[SIND(threadIdx.x, threadIdx.y - 1)] + shared_phi[SIND(threadIdx.x, threadIdx.y)]) / 2.f;


    float sumH = (xp*phi_xp + xm*phi_xm + yp*phi_yp + ym*phi_ym);
    float sumU =
        phi_xp * xp * (shared_flow_u[SIND(threadIdx.x + 1, threadIdx.y)] + shared_flow_du[SIND(threadIdx.x + 1, threadIdx.y)] - shared_flow_u[SIND(threadIdx.x, threadIdx.y)]) +
        phi_xm * xm * (shared_flow_u[SIND(threadIdx.x - 1, threadIdx.y)] + shared_flow_du[SIND(threadIdx.x - 1, threadIdx.y)] - shared_flow_u[SIND(threadIdx.x, threadIdx.y)]) +
        phi_yp * yp * (shared_flow_u[SIND(threadIdx.x, threadIdx.y + 1)] + shared_flow_du[SIND(threadIdx.x, threadIdx.y + 1)] - shared_flow_u[SIND(threadIdx.x, threadIdx.y)]) +
        phi_ym * ym * (shared_flow_u[SIND(threadIdx.x, threadIdx.y - 1)] + shared_flow_du[SIND(threadIdx.x, threadIdx.y - 1)] - shared_flow_u[SIND(threadIdx.x, threadIdx.y)]);
    float sumV =
        phi_xp * xp * (shared_flow_v[SIND(threadIdx.x + 1, threadIdx.y)] + shared_flow_dv[SIND(threadIdx.x + 1, threadIdx.y)] - shared_flow_v[SIND(threadIdx.x, threadIdx.y)]) +
        phi_xm * xm * (shared_flow_v[SIND(threadIdx.x - 1, threadIdx.y)] + shared_flow_dv[SIND(threadIdx.x - 1, threadIdx.y)] - shared_flow_v[SIND(threadIdx.x, threadIdx.y)]) +
        phi_yp * yp * (shared_flow_v[SIND(threadIdx.x, threadIdx.y + 1)] + shared_flow_dv[SIND(threadIdx.x, threadIdx.y + 1)] - shared_flow_v[SIND(threadIdx.x, threadIdx.y)]) +
        phi_ym * ym * (shared_flow_v[SIND(threadIdx.x, threadIdx.y - 1)] + shared_flow_dv[SIND(threadIdx.x, threadIdx.y - 1)] - shared_flow_v[SIND(threadIdx.x, threadIdx.y)]);

    float result_du =
      (shared_ksi[SIND(threadIdx.x, threadIdx.y)] * (-J13 - J12 * shared_flow_dv[SIND(threadIdx.x, threadIdx.y)]) + sumU) /
      (shared_ksi[SIND(threadIdx.x, threadIdx.y)] * J11 + sumH);

    float result_dv =
      (shared_ksi[SIND(threadIdx.x, threadIdx.y)] * (-J23 - J12 * result_du) + sumV) /
      (shared_ksi[SIND(threadIdx.x, threadIdx.y)] * J22 + sumH);

    //float result_dv =
    //    (shared_ksi[SIND(threadIdx.x, threadIdx.y)] * (-J23 - J12 * shared_flow_du[SIND(threadIdx.x, threadIdx.y)]) + sumV) /
    //    (shared_ksi[SIND(threadIdx.x, threadIdx.y)] * J22 + sumH);

    temp_du[IND(global_id.x, global_id.y)] = result_du;
    temp_dv[IND(global_id.x, global_id.y)] = result_dv;

  }
}