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

#ifndef GPUFLOW3D_CUDA_OPERATIONS_2D_CUDA_OPERATION_CONVOLUTION_2D_H_
#define GPUFLOW3D_CUDA_OPERATIONS_2D_CUDA_OPERATION_CONVOLUTION_2D_H_

#include <cuda.h>

#include "src/cuda_operations/cuda_operation_base.h"
#include "src/data_types/data_structs.h"

class CudaOperationConvolution2D : public CudaOperationBase {
private:
  CUfunction cuf_convolution_rows_;
  CUfunction cuf_convolution_cols_;

  DataSize3 dev_container_size_;

  float* kernel_ = nullptr;
  size_t kernel_length_;
  size_t kernel_radius_;

  void ComputeGaussianKernel(float sigma, size_t precision, float pixel_size);

  void ResampleRows(CUdeviceptr input, CUdeviceptr output, DataSize3& input_size, size_t pitch) const;
  void ResampleColumns(CUdeviceptr input, CUdeviceptr output, DataSize3& input_size, size_t pitch) const;

public:
    CudaOperationConvolution2D();

  bool Initialize(const OperationParameters* params = nullptr) override;
  void Execute(OperationParameters& params) override;
  void PrintConvolutionKernel();

  ~CudaOperationConvolution2D();
};

#endif // !GPUFLOW3D_CUDA_OPERATIONS_2D_CUDA_OPERATION_CONVOLUTION_2D_H_
