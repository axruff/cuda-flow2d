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

#ifndef GPUFLOW3D_OPTICAL_FLOW_SETTINGS_H_
#define GPUFLOW3D_OPTICAL_FLOW_SETTINGS_H_

#include <stdlib.h>
#include <string>
#include <map>
#include <vector>

#include "tinyxml.h"

using namespace std;

namespace OpticFlow {

	/**
	* Settings class, which contains all the settings for Optical flow framework
	*/
	class Settings
	{
	public:

		// Input settings
		string inputPath;
		string outputPath;
		string fileName1;
		string fileName2;

		// General
		int width;
		int height;
		float sigma;
		float precision;
		int medianRadius;

		// Solver settings
		int iterInner;
		int iterOuter;
		float alpha;

		float e_smooth;
		float e_data;

		int levels;
		float warpScale;

		float flowScale;

        bool press_key;


    private:
        string content;


	public:
		int LoadSettings(string fileName);
		void LoadSettingsManually();
	};

	



}

#endif // !GPUFLOW3D_OPTICAL_FLOW_SETTINGS_H_
