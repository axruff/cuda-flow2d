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

#include "io_utils.h"

#include <math.h>
#include <fstream>
#include <iostream>

#include "src/data_types/data2d.h"

using namespace IOUtils;




void IOUtils::WriteFlowToImageRGB(Data2D& u, Data2D& v, float flowMaxScale, string fileName)
{

    float maxLength = flowMaxScale;

    // Compute scaling factor
    float factor = 1.0 / maxLength;

    RGBColor rgb;

    fstream ofile(fileName.c_str(), ios::out | ios::binary);

    if (!ofile.is_open())
    {
        std::cerr << "Error: cannot save file " << std::endl;
        exit(255);
    }

    int nx = u.Width(); 
    int ny = u.Height();

    GRAY res;

    ofile<<"P6 \n";
    ofile<<nx <<" "<<ny<<" \n255\n";

    /* write image data */
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < nx; j++) {

            rgb =  ConvertToRGB(u.Data(j, i) * factor, v.Data(j, i) * factor);

            res = ConvertToGray(rgb.r);
            ofile.write((char *)&res, sizeof(GRAY));

            res = ConvertToGray(rgb.g);
            ofile.write((char *)&res, sizeof(GRAY));

            res = ConvertToGray(rgb.b);
            ofile.write((char *)&res, sizeof(GRAY));
        }
    }
    ofile.close();

}

void IOUtils::WriteMagnitudeToFileF32(Data2D& u, Data2D& v, string fileName)
{
    //char str[300];

    fstream ofile(fileName.c_str(), ios::out | ios::binary);

    if (!ofile.is_open())
    {
        std::cerr << "Error: cannot save file " << std::endl;
        exit(255);
    }

    int nx = u.Width();
    int ny = u.Height();


    float res;

    float * buf = new float[nx];

    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < nx; j++) {

            res = sqrt(u.Data(j, i)*u.Data(j, i) + v.Data(j, i)*v.Data(j, i));

            buf[j] = res;
        }
        ofile.write((char *)buf, nx * sizeof(float));
    }

    delete[] buf;

    ofile.close();
}

RGBColor::RGBColor()
{
    this->r = 0;
    this->g = 0;
    this->b = 0;
}

RGBColor::RGBColor(int r, int g, int b)
{
    this->r = r;
    this->g = g;
    this->b = b;
}


/*****************************************************************************/
/*                                                                           */
/*                   Copyright 08/2006 by Dr. Andres Bruhn                   */
/*     Faculty of Mathematics and Computer Science, Saarland University,     */
/*                           Saarbruecken, Germany.                          */
/*																			 */
/*						   Modified by Alexey Ershov		                 */
/*																			 */
/*****************************************************************************/
RGBColor IOUtils::ConvertToRGB(float x, float y)
{
    /********************************************************/
    float Pi;          /* pi                                                   */
    float amp;         /* amplitude (magnitude)                                */
    float phi;         /* phase (angle)                                        */
    float alpha, beta; /* weights for linear interpolation                     */
    /********************************************************/

    RGBColor rgb;

 /*   if (isUnknownFlow(x, y)) {
        x = 0.0;
        y = 0.0;
    }*/

    /* set pi */
    Pi = 2.0 * acos(0.0);

    /* determine amplitude and phase (cut amp at 1) */
    amp = sqrt(x * x + y * y);
    if (amp > 1) amp = 1;
    if (x == 0.0)
    if (y >= 0.0) phi = 0.5 * Pi;
    else phi = 1.5 * Pi;
    else if (x > 0.0)
    if (y >= 0.0) phi = atan(y/x);
    else phi = 2.0 * Pi + atan(y/x);
    else phi = Pi + atan(y/x);

    phi = phi / 2.0;

    // interpolation between red (0) and blue (0.25 * Pi)
    if ((phi >= 0.0) && (phi < 0.125 * Pi)) {
        beta  = phi / (0.125 * Pi);
        alpha = 1.0 - beta;
        rgb.r = (int)floor(amp * (alpha * 255.0 + beta * 255.0));
        rgb.g = (int)floor(amp * (alpha *   0.0 + beta *   0.0));
        rgb.b = (int)floor(amp * (alpha *   0.0 + beta * 255.0));
    }
    if ((phi >= 0.125 * Pi) && (phi < 0.25 * Pi)) {
        beta  = (phi-0.125 * Pi) / (0.125 * Pi);
        alpha = 1.0 - beta;
        rgb.r = (int)floor(amp * (alpha * 255.0 + beta *  64.0));
        rgb.g = (int)floor(amp * (alpha *   0.0 + beta *  64.0));
        rgb.b = (int)floor(amp * (alpha * 255.0 + beta * 255.0));
    }
    // interpolation between blue (0.25 * Pi) and green (0.5 * Pi)
    if ((phi >= 0.25 * Pi) && (phi < 0.375 * Pi)) {
        beta  = (phi - 0.25 * Pi) / (0.125 * Pi);
        alpha = 1.0 - beta;
        rgb.r = (int)floor(amp * (alpha *  64.0 + beta *   0.0));
        rgb.g = (int)floor(amp * (alpha *  64.0 + beta * 255.0));
        rgb.b = (int)floor(amp * (alpha * 255.0 + beta * 255.0));
    }
    if ((phi >= 0.375 * Pi) && (phi < 0.5 * Pi)) {
        beta  = (phi - 0.375 * Pi) / (0.125 * Pi);
        alpha = 1.0 - beta;
        rgb.r = (int)floor(amp * (alpha *   0.0 + beta *   0.0));
        rgb.g = (int)floor(amp * (alpha * 255.0 + beta * 255.0));
        rgb.b = (int)floor(amp * (alpha * 255.0 + beta *   0.0));
    }
    // interpolation between green (0.5 * Pi) and yellow (0.75 * Pi)
    if ((phi >= 0.5 * Pi) && (phi < 0.75 * Pi)) {
        beta  = (phi - 0.5 * Pi) / (0.25 * Pi);
        alpha = 1.0 - beta;
        rgb.r = (int)floor(amp * (alpha * 0.0   + beta * 255.0));
        rgb.g = (int)floor(amp * (alpha * 255.0 + beta * 255.0));
        rgb.b = (int)floor(amp * (alpha * 0.0   + beta * 0.0));
    }
    // interpolation between yellow (0.75 * Pi) and red (Pi)
    if ((phi >= 0.75 * Pi) && (phi <= Pi)) {
        beta  = (phi - 0.75 * Pi) / (0.25 * Pi);
        alpha = 1.0 - beta;
        rgb.r = (int)floor(amp * (alpha * 255.0 + beta * 255.0));
        rgb.g = (int)floor(amp * (alpha * 255.0 + beta *   0.0));
        rgb.b = (int)floor(amp * (alpha * 0.0   + beta *   0.0));
    }

    /* check RGBColor range */
    rgb.r = ConvertToByte(rgb.r);
    rgb.g = ConvertToByte(rgb.g);
    rgb.b = ConvertToByte(rgb.b);

    return rgb;
}

