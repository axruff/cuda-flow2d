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

#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>

#include <cuda.h>

#include "src/optical_flow/optical_flow_2d.h"

#include "src/correlation/correlation_flow_2d.h"
//#include "test_correlation.h"

#include "src/data_types/data2d.h"
#include "src/data_types/data_structs.h"
#include "src/data_types/operation_parameters.h"
#include "src/utils/cuda_utils.h"
#include "src/utils/io_utils.h"
#include "src/utils/settings.h"

using namespace OpticFlow;


int main(int argc, char** argv)
{


    /* Initialize CUDA */
    CUcontext cu_context;
    if (!InitCudaContextWithFirstAvailableDevice(&cu_context)) {
        return 1;
    }

    bool test_correlation = true;

    if (test_correlation) {
        std::printf("//----------------------------------------------------------------------//\n");
        std::printf("//        2D Correlation flow (Test) using NVIDIA CUDA. Version 0.5	   \n");
        std::printf("//                                                                        \n");
        std::printf("//           Author: Alexey Ershov. <ershov.alexey@gmail.com>             \n");
        std::printf("//            Karlsruhe Institute of Technology. 2009 - 2018              \n");
        std::printf("//----------------------------------------------------------------------//\n");

        const bool key_press = true;
        const bool use_visualization = false;
        const bool silent_mode = true;

        const size_t width = 128;
        const size_t height = 128;

        /* Correlation flow variables */
        size_t  correlation_window_size = 18;


        /*------------------------------------------------------*/
        /*               Correlation algorithm                  */
        /*------------------------------------------------------*/

        /* Correlation flow computation class */
        CorrelationFlow2D correlation_flow;

        Data2D image;
        DataSize3 image_size ={ width, height, 1 };

        /* Load input data */
        if (!image.ReadRAWFromFileF32("./data/real_frame-128-128.raw", image_size.width, image_size.height)) {
            //if (!image.ReadRAWFromFileU8("./data/squares_many.raw", image_size.width, image_size.height)) {
            //if (!image.ReadRAWFromFileF32("./data/73_flat_corr.raw", image_size.width, image_size.height)) {
            return 2;
        }


        if (correlation_flow.Initialize(image_size, correlation_window_size)) {



            Data2D flow_x(image_size.width, image_size.height);
            Data2D flow_y(image_size.width, image_size.height);
            Data2D corr(image_size.width, image_size.height);

            Data2D corr_temp(image_size.width*correlation_window_size, image_size.height*correlation_window_size);

            correlation_flow.silent = silent_mode;

            OperationParameters params;
            // params.PushValuePtr("correlation_window_size", &correlation_window_size);
            //params.PushValuePtr("warp_scale_factor", &warp_scale_factor);


            correlation_flow.ComputeFlow(image, flow_x, flow_y, corr, corr_temp, params);

            std::string filename =
                "-" + std::to_string(width) +
                "-" + std::to_string(height) + ".raw";

            std::string filename_ext =
                "-" + std::to_string(width*correlation_window_size) +
                "-" + std::to_string(height*correlation_window_size) + ".raw";

            flow_x.WriteRAWToFileF32(std::string("./data/output/corr-flow-x" + filename).c_str());
            flow_y.WriteRAWToFileF32(std::string("./data/output/corr-flow-y" + filename).c_str());
            corr.WriteRAWToFileF32(std::string("./data/output/corr-coeff" + filename).c_str());

            corr_temp.WriteRAWToFileF32(std::string("./data/output/corr-temp" + filename_ext).c_str());

            IOUtils::WriteFlowToImageRGB(flow_x, flow_y, 3, "./data/output/corr-res.pgm");



            correlation_flow.Destroy();
        }

        if (key_press) {
            std::printf("Press enter to continue...");
            std::getchar();
        }


        /* Release resources */
        cuCtxDestroy(cu_context);

        return 0;
        
    }

    std::printf("//----------------------------------------------------------------------//\n");
    std::printf("//            2D Optical flow using NVIDIA CUDA. Version 0.5.0	        //\n");
    std::printf("//                                                                      //\n");
    std::printf("//            Karlsruhe Institute of Technology. 2015 - 2018            //\n");
    std::printf("//----------------------------------------------------------------------//\n");

  const bool key_press = true;
  const bool use_visualization = false;
  const bool silent_mode = true;


  /* Dataset variables */
  const size_t width = 584;
  const size_t height = 388;


  /* Optical flow variables */
  size_t  warp_levels_count       = 20;
  float   warp_scale_factor       = 0.9f;
  size_t  outer_iterations_count  = 40;
  size_t  inner_iterations_count  = 5;
  float   equation_alpha          = 3.5f;
  float   equation_smoothness     = 0.001f;
  float   equation_data           = 0.001f;
  size_t  median_radius           = 5;
  float   gaussian_sigma          = 2.5f;

  string  file_name1              = "rub1.raw";
  string  file_name2              = "rub2.raw";
  string  input_path              =  "./data/";
  string  output_path             =  "./data/output/";


  /* Optical flow computation class */
  OpticalFlow2D optical_flow;

  
  Data2D frame_0;
  Data2D frame_1;
  DataSize3 data_size ={ width, height, 1 };

  /* Read settings */
  string settingsPath = "settings.xml";

  if (argc > 2)  {
      cout<<"usage: "<< argv[0] <<" <settings file>. Otherwise settings.xml in the current directory is used"<<endl;
  }
  else if (argc != 1) {
      string settingsFile = (argc == 1) ? settingsPath : string(argv[1]);

      // Create Settings class
      cout<<"Reading settings: "<<settingsFile<<endl;;
      Settings settings = Settings();

      int settingsError = 0;

      try {
          settingsError = settings.LoadSettings(settingsFile);
      }
      catch (...)
      {
          cout<<"Unexpected error reading settings file"<<endl;
          settingsError = 1;
      }
      if (settingsError) {
          cout<<"TERMINATING. Error reading settings: "<<settingsFile<<endl;
          return 3;
      }
      else
          cout<<"OK"<<endl<<endl;

      warp_levels_count = settings.levels;
      warp_scale_factor = settings.warpScale;
      outer_iterations_count = settings.iterOuter;
      inner_iterations_count = settings.iterInner;
      equation_alpha = settings.alpha;
      equation_data = settings.e_data;
      equation_smoothness = settings.e_smooth;
      median_radius = settings.medianRadius;
      gaussian_sigma = settings.sigma;

    
  }


  /* Load input data */
  if (!frame_0.ReadRAWFromFileU8("./data/rub1.raw", data_size.width, data_size.height) ||
      !frame_1.ReadRAWFromFileU8("./data/rub2.raw", data_size.width, data_size.height)) {
      return 2;
  }


  if (optical_flow.Initialize(data_size)) {

    Data2D flow_u(data_size.width, data_size.height);
    Data2D flow_v(data_size.width, data_size.height);

    optical_flow.silent = silent_mode;

    OperationParameters params;
    params.PushValuePtr("warp_levels_count",      &warp_levels_count);
    params.PushValuePtr("warp_scale_factor",      &warp_scale_factor);
    params.PushValuePtr("outer_iterations_count", &outer_iterations_count);
    params.PushValuePtr("inner_iterations_count", &inner_iterations_count);
    params.PushValuePtr("equation_alpha",         &equation_alpha);
    params.PushValuePtr("equation_smoothness",    &equation_smoothness);
    params.PushValuePtr("equation_data",          &equation_data);
    params.PushValuePtr("median_radius",          &median_radius);
    params.PushValuePtr("gaussian_sigma",         &gaussian_sigma);

    optical_flow.ComputeFlow(frame_0, frame_1, flow_u, flow_v, params);
    
    std::string filename =
      "-" + std::to_string(width) +
      "-" + std::to_string(height) + ".raw";

    flow_u.WriteRAWToFileF32(std::string("./data/output/flow-u" + filename).c_str());
    flow_v.WriteRAWToFileF32(std::string("./data/output/flow-v" + filename).c_str());

    IOUtils::WriteFlowToImageRGB(flow_u, flow_v, 3, "./data/output/res.pgm");



    optical_flow.Destroy();
  }

  if (key_press) {
    std::printf("Press enter to continue...");
    std::getchar();
  }



  /* Release resources */
  cuCtxDestroy(cu_context);

  return 0;
}