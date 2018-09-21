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

#ifndef GPUFLOW3D_CORRELATION_CORRELATION_FLOW_2D_H_
#define GPUFLOW3D_CORRELATION_CORRELATION_FLOW_2D_H_

#include <forward_list>
#include <stack>

#include <cuda.h>

#include "src/cuda_operations/cuda_operation_base.h"
#include "src/cuda_operations/2d/cuda_operation_correlation_2d.h"

#include "src/data_types/operation_parameters.h"
#include "src/data_types/data2d.h"


class CorrelationFlow2D {
private:
    const char* name_ = nullptr;
    bool initialized_ = false;

    CUdeviceptr dev_container_extended_corr;
    CUdeviceptr dev_container_extended_corr_max;

    const size_t dev_containers_count_ = 4;
    
    size_t correlation_window_size_ = -1;

    DataSize3 dev_container_size_;
    DataSize3 dev_container_extended_size_;

    std::forward_list<CudaOperationBase*> cuda_operations_;
    std::stack<CUdeviceptr> cuda_memory_ptrs_;

    CudaOperationCorrelation2D cuop_correlation_;


    bool InitCudaOperations();
    bool InitCudaMemory();

    bool IsInitialized() const;

public:
    CorrelationFlow2D();

    const char* GetName() const;

    bool Initialize(const DataSize3& data_size, const size_t correlation_window_size);
    void ComputeFlow(Data2D& image, Data2D& flow_x, Data2D& flow_y, Data2D& corr, Data2D& corr_temp, OperationParameters& params);
    void Destroy();

    bool silent = false;

    ~CorrelationFlow2D();
};


#endif // !GPUFLOW3D_CORRELATION_CORRELATION_FLOW_2D_H_