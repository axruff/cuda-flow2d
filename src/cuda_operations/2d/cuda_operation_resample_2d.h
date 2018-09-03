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

#ifndef GPUFLOW3D_CUDA_OPERATIONS_2D_CUDA_OPERATION_RESAMPLE_2D_H_
#define GPUFLOW3D_CUDA_OPERATIONS_2D_CUDA_OPERATION_RESAMPLE_2D_H_

#include <cuda.h>

#include "src/cuda_operations/cuda_operation_base.h"
#include "src/data_types/data_structs.h"

class CudaOperationResample2D : public CudaOperationBase {
private: 
  CUfunction cuf_resample_x_;
  CUfunction cuf_resample_y_;

  void ResampleX(CUdeviceptr input, CUdeviceptr output, DataSize3& input_size, DataSize3& output_size) const;
  void ResampleY(CUdeviceptr input, CUdeviceptr output, DataSize3& input_size, DataSize3& output_size) const;


public:
  CudaOperationResample2D();

  bool Initialize(const OperationParameters* params = nullptr) override;
  void Execute(OperationParameters& params) override;
};

#endif // !GPUFLOW3D_CUDA_OPERATIONS_2D_CUDA_OPERATION_RESAMPLE_2D_H_
