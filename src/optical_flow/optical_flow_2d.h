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

#ifndef GPUFLOW3D_OPTICAL_FLOW_OPTICAL_FLOW_2D_H_
#define GPUFLOW3D_OPTICAL_FLOW_OPTICAL_FLOW_2D_H_

#include <forward_list>
#include <stack>

#include <cuda.h>

#include "src/cuda_operations/cuda_operation_base.h"
#include "src/cuda_operations/2d/cuda_operation_add_2d.h"
#include "src/cuda_operations/2d/cuda_operation_median_2d.h"
#include "src/cuda_operations/2d/cuda_operation_registration_2d.h"
#include "src/cuda_operations/2d/cuda_operation_resample_2d.h"
#include "src/cuda_operations/2d/cuda_operation_solve_2d.h"

#include "src/data_types/operation_parameters.h"
#include "src/data_types/data2d.h"

#include "src/optical_flow/optical_flow_base_2d.h"

class OpticalFlow2D : public OpticalFlowBase2D{
private:
  const size_t dev_containers_count_ = 12;
  DataSize3 dev_container_size_;

  std::forward_list<CudaOperationBase*> cuda_operations_;
  std::stack<CUdeviceptr> cuda_memory_ptrs_;

  CudaOperationAdd2D cuop_add_;
  CudaOperationMedian2D cuop_median_;
  CudaOperationRegistration2D cuop_register_;
  CudaOperationResample2D cuop_resample_;
  CudaOperationSolve2D cuop_solve_;

  bool InitCudaOperations();
  bool InitCudaMemory();

public:
    OpticalFlow2D();

  bool Initialize(const DataSize3& data_size) override;
  void ComputeFlow(Data2D& frame_0, Data2D& frame_1, Data2D& flow_u, Data2D& flow_v, OperationParameters& params) override;
  void Destroy() override;

  bool silent = false;

  ~OpticalFlow2D() override;
};


#endif // !GPUFLOW3D_OPTICAL_FLOW_OPTICAL_FLOW_2D_H_
