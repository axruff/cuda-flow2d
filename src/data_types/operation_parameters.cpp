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

#include "src/data_types/operation_parameters.h"

OperationParameters::OperationParameters()
{
}

bool OperationParameters::PushValuePtr(std::string key, void* value_ptr)
{
  if (!map_.count(key)) {
    map_.insert({ key, value_ptr });
    return true;
  }
  return false;
}

void* OperationParameters::GetValuePtr(std::string key) const
{
  if (map_.count(key)) {
    return map_.at(key);
  }
  return nullptr;
}

void OperationParameters::Clear()
{
  map_.clear();
}