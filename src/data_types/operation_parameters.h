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

#ifndef GPUFLOW3D_DATA_TYPES_OPERATION_PARAMETERS_H_
#define GPUFLOW3D_DATA_TYPES_OPERATION_PARAMETERS_H_

#include <string>
#include <unordered_map>

class OperationParameters {
private:
  std::unordered_map<std::string, void*> map_;

public:
  OperationParameters();

  bool PushValuePtr(std::string key, void* value_ptr);
  void* GetValuePtr(std::string key) const;
  void Clear();
};

#endif // !GPUFLOW3D_DATA_TYPES_OPERATION_PARAMETERS_H_
