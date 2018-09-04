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

#include "src/data_types/data3d.h"

#include <algorithm>
#include <cstdio>
#include <climits>
#include <cstring>

#include <cuda.h>

#include "src/utils/cuda_utils.h"



//#define ALLOCATE_PINNED_MEMORY

Data3D::Data3D()
  : width_(0), height_(0), depth_(0)
{}

Data3D::Data3D(size_t width, size_t height, size_t depth)
  : width_(width), height_(height), depth_(depth)
{
  data_ = static_cast<float*>(AllocateMemory(width_, height_, depth_));
}

void Data3D::Swap(Data3D& data3d)
{
  if (this->width_ == data3d.width_ &&
      this->height_ == data3d.height_ &&
      this->depth_ == data3d.depth_) {
    std::swap(this->data_, data3d.data_);
  } else {
    std::printf("Error. Cannot swap two Data3D objects (wrong dimensions).\n");
  }
}

void* Data3D::AllocateMemory(size_t width, size_t height, size_t depth)
{
  void* ptr = nullptr;

#ifdef ALLOCATE_PINNED_MEMORY
  if (CheckCudaError(cuMemAllocHost(&ptr, width * height * depth * sizeof(float)))) {
    Invalidate();
    return nullptr;
  }
#else
  try {
    ptr = new float[width * height * depth];
  }
  catch (std::bad_alloc& ba) {
    Invalidate();
    std::printf("Error. Cannot allocate memory on the host. (%s)\n", ba.what());
  }
#endif

  return ptr;
}

void Data3D::FreeMemory(void* pointer)
{
#ifdef ALLOCATE_PINNED_MEMORY
  if (pointer)
    CheckCudaError(cuMemFreeHost(pointer));
#else
  if (pointer) {
    delete[] pointer;
    pointer = nullptr;
  }
#endif
}

void Data3D::ZeroData()
{
  if (data_) {
    std::memset(data_, 0, width_ * height_ * depth_ * sizeof(float));
  }
}

bool Data3D::ReadRAWFromFileU8(const char* filename, size_t width, size_t height, size_t depth)
{
  bool loaded = false;
  std::FILE *file = std::fopen(filename, "rb");
  if (file) {
    this->Invalidate();

    width_ = width;
    height_ = height;
    depth_ = depth;
    data_ = static_cast<float*>(AllocateMemory(width_, height_, depth_));

    if (data_) {
      unsigned char *dataU8 = new unsigned char[width_];

      for (size_t z = 0; z < depth_; ++z) {
        for (size_t y = 0; y < height_; ++y) {
          size_t readed = std::fread(dataU8, sizeof(unsigned char), width_, file);

          if (readed == width_) {
            for (size_t x = 0; x < width_; ++x) {
              data_[Index(x, y, z)] = static_cast<float>(dataU8[x]);
            }
          } else {
            goto loop_break;
          }
        }
      }

      if (std::fread(dataU8, sizeof(unsigned char), width_, file) == 0) {
        loaded = true;
      } else {

      loop_break:
        std::printf("Error reading RAW data from file '%s': wrong dimensions.", filename);
        this->Invalidate();
      }

      delete[] dataU8;
      std::fclose(file);
    }
  } else {
    std::printf("Cannot open file '%s'.\n", filename);
  }
  return loaded;
}

bool Data3D::ReadRAWFromFileF32(const char* filename, size_t width, size_t height, size_t depth)
{
  bool loaded = false;
  std::FILE *file = std::fopen(filename, "rb");
  if (file) {
    this->Invalidate();

    width_ = width;
    height_ = height;
    depth_ = depth;
    data_ = static_cast<float*>(AllocateMemory(width_, height_, depth_));

    if (data_) {
      for (size_t z = 0; z < depth_; ++z) {
        for (size_t y = 0; y < height_; ++y) {
          size_t readed = std::fread(&data_[Index(0, y, z)], sizeof(float), width_, file);
          if (readed != width_) {
            goto loop_break;
          }
        }
      }

      if (std::fread(data_, sizeof(unsigned char), width_, file) == 0) {
        loaded = true;
      } else {
      loop_break:
        std::printf("Error reading RAW data from file '%s': wrong dimensions.", filename);
        this->Invalidate();
      }

      std::fclose(file);
    }
  } else {
    std::printf("Cannot open file '%s'.\n", filename);
  }
  return loaded;
}

bool Data3D::WriteRAWToFileU8(const char* filename)
{
  std::FILE *file = std::fopen(filename, "wb");
  if (file) {
   unsigned char *dataU8 = new unsigned char[width_];

    for (size_t z = 0; z < depth_; ++z) {
      for (size_t y = 0; y < height_; ++y) {
        for (size_t x = 0; x < width_; ++x) {
          float dataF32 = std::min(255.f, std::max(0.f, data_[Index(x, y, z)]));
          dataU8[x] = static_cast<unsigned char>(dataF32);
        }
        size_t written = std::fwrite(dataU8, sizeof(unsigned char), width_, file);
        if (written != width_) {
          std::printf("Error writing RAW data to file '%s'.", filename);
          delete[] dataU8;
          std::fclose(file);
          return false;
        }
      }
    }
    delete[] dataU8;
    std::fclose(file);
    return true;
  } else {
    std::printf("Cannot open file '%s'.\n", filename);
    return false;
  }
}

bool Data3D::WriteRAWToFileF32(const char* filename)
{
  std::FILE *file = std::fopen(filename, "wb");
  if (file) {
   unsigned char *dataU8 = new unsigned char[width_];

    for (size_t z = 0; z < depth_; ++z) {
      for (size_t y = 0; y < height_; ++y) {
        size_t written = std::fwrite(&data_[Index(0, y, z)], sizeof(float), width_, file);
        if (written != width_) {
          std::printf("Error writing RAW data to file '%s'.", filename);
          std::fclose(file);
          return false;
        }
      }
    }
    std::fclose(file);
    return true;
  } else {
    std::printf("Cannot open file '%s'.\n", filename);
    return false;
  }
}

bool Data3D::WriteFlowToFileVTK(const char* filename, const Data3D& flow_u, const Data3D& flow_v, const Data3D& flow_w)
{
  std::FILE *file = std::fopen(filename, "wb");
  if (file) {
    std::fprintf(file, "# vtk DataFile Version 2.0\n");
    std::fprintf(file, "3D Vector field computed by GpuFlow3D\n");
    std::fprintf(file, "BINARY\n");
    std::fprintf(file, "DATASET STRUCTURED_POINTS\n");
    std::fprintf(file, "DIMENSIONS %d %d %d\n", flow_u.width_, flow_u.height_, flow_u.depth_);
    std::fprintf(file, "ORIGIN 0 0 0\n");
    std::fprintf(file, "SPACING 1 1 1\n");
    std::fprintf(file, "POINT_DATA %d\n", flow_u.width_ * flow_u.height_ * flow_u.depth_);
    std::fprintf(file, "VECTORS vectors float\n");

    for (size_t z = 0; z < flow_u.depth_; ++z) {
      for (size_t y = 0; y < flow_u.height_; ++y) {
        for (size_t x = 0; x < flow_u.width_; ++x) {
          std::fwrite(&flow_u.data_[(z * flow_u.height_ + y) * flow_u.width_ + x], sizeof(float), 1, file);
          std::fwrite(&flow_v.data_[(z * flow_v.height_ + y) * flow_v.width_ + x], sizeof(float), 1, file);
          std::fwrite(&flow_w.data_[(z * flow_w.height_ + y) * flow_w.width_ + x], sizeof(float), 1, file);
        }
      }
    }

    std::fclose(file);
    return true;
  } else {
    std::printf("Cannot open file '%s'.\n", filename);
    return false;
  }
}

void Data3D::Invalidate()
{
  width_ = 0;
  height_ = 0;
  depth_ = 0;
  FreeMemory(data_);
}

Data3D::~Data3D()
{
  FreeMemory(data_);
}