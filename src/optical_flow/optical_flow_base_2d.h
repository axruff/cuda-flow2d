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

#ifndef GPUFLOW3D_OPTICAL_FLOW_OPTICAL_FLOW_BASE_2D_H_
#define GPUFLOW3D_OPTICAL_FLOW_OPTICAL_FLOW_BASE_2D_H_

#include "src/data_types/data2d.h"
#include "src/data_types/data_structs.h"
#include "src/data_types/operation_parameters.h"

class OpticalFlowBase2D {
private:
  const char* name_ = nullptr;

protected:
  bool initialized_ = false;

  DataConstancy data_constancy_;

  OpticalFlowBase2D(const char* name);

  size_t GetMaxWarpLevel(size_t width, size_t height, float scale_factor) const;

  bool IsInitialized() const;

public:
  const char* GetName() const;

  virtual bool Initialize(const DataSize3& data_size, DataConstancy data_constancy = DataConstancy::Grey) = 0;
  virtual void ComputeFlow(Data2D& frame_0, Data2D& frame_1, Data2D& flow_u, Data2D& flow_v, OperationParameters& params);
  virtual void Destroy();

  virtual ~OpticalFlowBase2D();
};



#endif // !GPUFLOW3D_OPTICAL_FLOW_OPTICAL_FLOW_BASE_2D_H_
