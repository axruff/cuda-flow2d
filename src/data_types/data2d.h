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

#ifndef GPUFLOW3D_DATA_TYPES_DATA2D_H_
#define GPUFLOW3D_DATA_TYPES_DATA2D_H_

#include <cstdlib>

class Data2D
{
private:
  float   *data_ = nullptr;
  size_t  width_;
  size_t  height_;

  inline size_t Index(size_t x, size_t y) {
    return y * width_ + x;
  }
  
  void Invalidate();

  void* AllocateMemory(size_t width, size_t height);
  void FreeMemory(void* pointer);

public:
  Data2D();
  Data2D(size_t width, size_t height);
  
  inline size_t Width() { return width_; };
  inline size_t Height() { return height_; };
  inline float* DataPtr() { return data_; };
  inline float& Data(size_t x, size_t y) { return data_[Index(x, y)]; };

  void Swap(Data2D& data2d);

  void ZeroData();

  bool ReadRAWFromFileU8(const char* filename, size_t width, size_t height);
  bool ReadRAWFromFileF32(const char* filename, size_t width, size_t height);

  bool WriteRAWToFileU8(const char* filename);
  bool WriteRAWToFileF32(const char* filename);
  
  //static bool WriteFlowToFileVTK(const char* filename, const Data3D& flow_u, const Data3D& flow_v, const Data3D& flow_w);

  ~Data2D();
};



#endif // !GPUFLOW3D_DATA_TYPES_DATA2D_H_