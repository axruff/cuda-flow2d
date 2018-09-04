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

#include "src/optical_flow/optical_flow_base_2d.h"

#include <cmath>

OpticalFlowBase2D::OpticalFlowBase2D(const char* name)
  : name_(name)
{
}

const char* OpticalFlowBase2D::GetName() const
{
  return name_;
}

size_t OpticalFlowBase2D::GetMaxWarpLevel(size_t width, size_t height, float scale_factor) const
{
    /* Compute maximum number of warping levels for given image size and warping reduction factor */
    size_t r_width = 1;
    size_t r_height = 1;
    size_t level_counter = 1;

    while (scale_factor < 1.f) {
        float scale = std::pow(scale_factor, static_cast<float>(level_counter));
        r_width = static_cast<size_t>(std::ceil(width * scale));
        r_height = static_cast<size_t>(std::ceil(height * scale));

        if (r_width < 4 || r_height < 4 ) {
            break;
        }
        ++level_counter;
    }

    if (r_width == 1 || r_height == 1) {
        --level_counter;
    }

    return level_counter;
}

bool OpticalFlowBase2D::IsInitialized() const
{
  if (!initialized_) {
    std::printf("Error: '%s' was not initialized.\n", name_);
  }
  return initialized_;
}

void OpticalFlowBase2D::ComputeFlow(Data2D& frame_0, Data2D& frame_1, Data2D& flow_u, Data2D& flow_v, OperationParameters& params)
{
  std::printf("Warning: '%s' ComputeFlow() was not defined.\n", name_);

}

void OpticalFlowBase2D::Destroy()
{
  initialized_ = false;
}

OpticalFlowBase2D::~OpticalFlowBase2D()
{
  if (initialized_) {
    Destroy();
  }
}
