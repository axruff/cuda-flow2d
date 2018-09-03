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

#ifndef GPUFLOW3D_UTILS_COMMON_UTILS_H_
#define GPUFLOW3D_UTILS_COMMON_UTILS_H_

#include <cstdio>

namespace Utils {

  void GetExecutablePath(char* path, size_t size);
  void PrintProgressBar(float complete);

} // namespace Utils

#define GET_PARAM_OR_RETURN(P, T, V, N)                                          \
  do {                                                                           \
    void* v_ptr = (P).GetValuePtr((N));                                          \
    if (v_ptr) {                                                                 \
      (V) = *(static_cast<T*>(v_ptr));                                           \
    } else {                                                                     \
      std::printf("Operation: '%s'. Missing parameter '%s'.\n", GetName(), (N)); \
      return;                                                                    \
    }                                                                            \
  } while (0)

#define GET_PARAM_OR_RETURN_VALUE(P, T, V, N, R)                                 \
  do {                                                                           \
    void* v_ptr = (P).GetValuePtr((N));                                          \
    if (v_ptr) {                                                                 \
      (V) = *(static_cast<T*>(v_ptr));                                           \
    } else {                                                                     \
      std::printf("Operation: '%s'. Missing parameter '%s'.\n", GetName(), (N)); \
      return (R);                                                                \
    }                                                                            \
  } while (0)

#define GET_PARAM_PTR_OR_RETURN(P, T, PTR, N)                                          \
  do {                                                                           \
    void* v_ptr = (P).GetValuePtr((N));                                          \
    if (v_ptr) {                                                                 \
      (PTR) = (static_cast<T*>(v_ptr));                                           \
    } else {                                                                     \
      std::printf("Operation: '%s'. Missing parameter '%s'.\n", GetName(), (N)); \
      return;                                                                    \
    }                                                                            \
  } while (0)

#endif // !GPUFLOW3D_UTILS_COMMON_UTILS_H_
