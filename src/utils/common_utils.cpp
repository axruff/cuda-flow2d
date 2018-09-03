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

#include "common_utils.h"

#include <cstring>

#ifdef _WIN32
#include <Windows.h>
#endif
#ifdef __gnu_linux__
#include <unistd.h>
#endif

namespace Utils {

  void GetExecutablePath(char* path, size_t size)
  {
#ifdef _WIN32
    GetModuleFileName(NULL, path, size);
    char* last_char = std::strrchr(path, '\\');
    if (last_char) {
      *last_char = '\0';
    }
#endif
#ifdef __gnu_linux__
    ssize_t link_size = readlink("/proc/self/exe", path, size);
    path[link_size] = '\0';
    char* last_char = std::strrchr(path, '/');
    if (last_char) {
      *last_char = '\0';
    }
#endif
  }

  void PrintProgressBar(float complete)
  {
    const size_t width = 40;
    char buffer[80];
    char* buffer_ptr = buffer;

    *(buffer_ptr++) = '[';
    for (size_t i = 0; i < width; ++i) {
      *(buffer_ptr++) = (i / static_cast<float>(width) < complete) ? '=' : ' ';
    }
    *(buffer_ptr++) = ']';
    *(buffer_ptr++) = '\0';

    printf("\r%s", buffer);
  }

} // namespace Utils