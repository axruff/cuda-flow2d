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

#ifndef GPUFLOW3D_UTILS_IO_UTILS_H_
#define GPUFLOW3D_UTILS_IO_UTILS_H_

#include "src/data_types/data2d.h"

#include <fstream>
#include <iostream>


using namespace std;

namespace IOUtils {

    typedef unsigned char GRAY;

    struct RGBColor
    {
        int r;
        int g;
        int b;

        RGBColor();

        RGBColor(int r, int g, int b) ;
    };


    //------------------------------------------------------------------
    // 2D Writing Routines
    //------------------------------------------------------------------
    void WriteFlowToImageRGB(Data2D& u, Data2D& v, float flowMaxScale, string fileName);
    void WriteMagnitudeToFileF32(Data2D& u, Data2D& v, string fileName);

    // Data convertion routines
    RGBColor ConvertToRGB(float x, float y);

    inline int ConvertToByte(int num)
    {
        return (num >=255) * 255 + ((num < 255) && (num > 0))* num;
    }

    inline GRAY ConvertToGray(float number)
    {
        GRAY result;

        if (number < 0.0)
            result = (GRAY)(0.0);
        else if (number > 255.0)
            result = (GRAY)(255.0);
        else
            result = (GRAY)(number);

        return result;

    }

}

#endif // !GPUFLOW3D_UTILS_IO_UTILS_H_

