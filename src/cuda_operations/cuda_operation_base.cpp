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

#include "cuda_operation_base.h"

#include <cstdio>
#include <cstring>

#include "src/utils/cuda_utils.h"

CudaOperationBase::CudaOperationBase(const char* name)
  : name_(name) 
{
}

const char* CudaOperationBase::GetName() const
{
  return name_;
};

bool CudaOperationBase::IsInitialized() const
{
  if (!initialized_) {
    std::printf("Error: Operation '%s' was not initialized.\n", name_);
  }
  return initialized_;
}

void CudaOperationBase::Execute(OperationParameters& params)
{
  std::printf("Warning: '%s' Execute() was not defined.\n", name_);
}

void CudaOperationBase::Destroy()
{
  if (cu_module_) {
    CheckCudaError(cuModuleUnload(cu_module_));
    cu_module_ = nullptr;
  }
  initialized_ = false;
}

CudaOperationBase::~CudaOperationBase()
{
  if (initialized_) {
    Destroy();
  }
}

