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

#include "correlation_flow_2d.h"

#include <iostream>

#include <cuda.h>

#include "src/utils/common_utils.h"
#include "src/utils/cuda_utils.h"



using namespace std;

CorrelationFlow2D::CorrelationFlow2D() :name_{ "Correlation Flow 2D Single GPU"}
{
    cuda_operations_.push_front(&cuop_add_);
    cuda_operations_.push_front(&cuop_correlation_);

}

bool CorrelationFlow2D::Initialize(const DataSize3& data_size, const size_t correlation_window_size)
{
    dev_container_size_ = data_size;
    dev_container_size_.pitch = 0;

    correlation_window_size_ = correlation_window_size;
    dev_container_extended_size_.height = dev_container_size_.height*correlation_window_size_;
    dev_container_extended_size_.width = dev_container_size_.width*correlation_window_size_;
    dev_container_extended_size_.pitch = 0;

    initialized_ = InitCudaMemory() && InitCudaOperations();
    return initialized_;
}


const char* CorrelationFlow2D::GetName() const
{
    return name_;
}

bool CorrelationFlow2D::IsInitialized() const
{
    if (!initialized_) {
        std::printf("Error: '%s' was not initialized.\n", name_);
    }
    return initialized_;
}

bool CorrelationFlow2D::InitCudaOperations()
{
    if (dev_container_size_.pitch == 0) {
        std::printf("Initialization failed. Device pitch is 0.\n");
        return false;
    }

    std::printf("Initialization of cuda operations...\n");

    OperationParameters op;
    op.PushValuePtr("container_size", &dev_container_size_);

    for (CudaOperationBase* cuop : cuda_operations_) {
        std::printf("%-18s: ", cuop->GetName());
        bool result = cuop->Initialize(&op);
        if (result) {
            std::printf("OK\n");
        }
        else {
            Destroy();
            return false;
        }
    }
    return true;
}

bool CorrelationFlow2D::InitCudaMemory()
{
    std::printf("Allocating memory on the device...\n");
    /* Check available memory on the cuda device */
    size_t free_memory;
    size_t total_memory;

    CheckCudaError(cuMemGetInfo(&free_memory, &total_memory));

    std::printf("Available\t:\t%.0fMB / %.0fMB\n",
        free_memory / static_cast<float>(1024 * 1024),
        total_memory / static_cast<float>(1024 * 1024));

    /* Estimate needed memory (approx.)*/
    int alignment = 512;
    CUdevice cu_device;
    CheckCudaError(cuCtxGetDevice(&cu_device));
    CheckCudaError(cuDeviceGetAttribute(&alignment, CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT, cu_device));
    alignment /= sizeof(float);

    /* Calculate pitch size for coalesed memory access*/
    size_t pitch = ((dev_container_size_.width % alignment == 0) ?
        (dev_container_size_.width) :
        (dev_container_size_.width + alignment - (dev_container_size_.width % alignment))) *
        sizeof(float);

    /* Pitch for extended correlation containers */
    size_t pitch_extended = (((dev_container_size_.width*correlation_window_size_) % alignment == 0) ?
        (dev_container_size_.width*correlation_window_size_) :
        (dev_container_size_.width*correlation_window_size_ + alignment - ((dev_container_size_.width*correlation_window_size_) % alignment))) *
        sizeof(float);

    cout<<"Width: "<< dev_container_size_.width<<endl;
    cout<<"Pitch: "<< pitch / sizeof(float)<<endl<<endl;

    cout<<"Extended Width: "<< dev_container_extended_size_.width<<endl;
    cout<<"Extended Pitch: "<< pitch_extended / sizeof(float)<<endl;

    size_t extended_buffers_count = 2;

    size_t needed_memory = pitch * dev_container_size_.height * dev_containers_count_;
    needed_memory +=  pitch_extended * dev_container_extended_size_.height * extended_buffers_count;


    std::printf("Needed (approx.):\t%.0fMB\n", needed_memory / static_cast<float>(1024 * 1024));

    size_t allocated_memory = 0;
    if (needed_memory < free_memory) {
        for (size_t i = 0; i < dev_containers_count_; ++i) {
            CUdeviceptr dev_container;
            bool error = CheckCudaError(cuMemAllocPitch(&dev_container,
                &pitch,
                dev_container_size_.width * sizeof(float),
                dev_container_size_.height * 1,
                sizeof(float)));

            /* Pitch should be the same for all containers */
            if (error || (i != 0 && dev_container_size_.pitch != pitch)) {
                std::printf("Error during device memory allocation.");
                Destroy();
                return false;
            }
            else {
                size_t container_size;
                CheckCudaError(cuMemGetAddressRange(NULL, &container_size, dev_container));
                allocated_memory += container_size;

                cuda_memory_ptrs_.push(dev_container);
                dev_container_size_.pitch = pitch;
            }
        }

        // Allocated memory for extended buffers

        bool error1 = CheckCudaError(cuMemAllocPitch(&dev_container_extended_corr,
            &pitch_extended,
            dev_container_extended_size_.width * sizeof(float),
            dev_container_extended_size_.height * 1,
            sizeof(float)));

        bool error2 = CheckCudaError(cuMemAllocPitch(&dev_container_extended_corr_max,
            &pitch_extended,
            dev_container_extended_size_.width * sizeof(float),
            dev_container_extended_size_.height * 1,
            sizeof(float)));

        if (error1 || error2) {
            std::printf("Error during device memory allocation: Extended buffers");
            Destroy();
            return false;

        }
        else {
            size_t container_size;
            CheckCudaError(cuMemGetAddressRange(NULL, &container_size, dev_container_extended_corr));
            allocated_memory += container_size;

            CheckCudaError(cuMemGetAddressRange(NULL, &container_size, dev_container_extended_corr_max));
            allocated_memory += container_size;

            dev_container_extended_size_.pitch = pitch_extended;

            cout<<"Device containers:"<<endl;
            cout<<dev_container_extended_corr<<endl;
            cout<<dev_container_extended_corr_max<<endl;

        }

        std::printf("Allocated\t:\t%.0fMB\n", allocated_memory / static_cast<float>(1024 * 1024));
        return true;
    }
    return false;
}


void CorrelationFlow2D::ComputeFlow(Data2D& image, Data2D& flow_x, Data2D& flow_y, Data2D& corr, Data2D& corr_temp, OperationParameters& params)
{
    if (!IsInitialized()) {
        return;
    }

    /* Correlation flow algorithm's parameters */
    //size_t  warp_levels_count;
    //float   warp_scale_factor;


    /* Lambda function for correct working of GET_PARAM_OR_RETURN */
    //GET_PARAM_OR_RETURN(params, size_t, correlation_window_size_, "correlation_window_size");
    //GET_PARAM_OR_RETURN(params, float, warp_scale_factor, "warp_scale_factor");


    std::printf("\nStarting correlation flow computation...\n");

    /* Create CUDA event for the time measure */
    CUevent cu_event_start;
    CUevent cu_event_stop;

    CheckCudaError(cuEventCreate(&cu_event_start, CU_EVENT_DEFAULT));
    CheckCudaError(cuEventCreate(&cu_event_stop, CU_EVENT_DEFAULT));

    CheckCudaError(cuEventRecord(cu_event_start, NULL));


    /* Copy input data to the device */
    CUdeviceptr dev_image = cuda_memory_ptrs_.top();
    cuda_memory_ptrs_.pop();
    CUdeviceptr dev_flow_x = cuda_memory_ptrs_.top();
    cuda_memory_ptrs_.pop();
    CUdeviceptr dev_flow_y = cuda_memory_ptrs_.top();
    cuda_memory_ptrs_.pop();
    CUdeviceptr dev_corr = cuda_memory_ptrs_.top();
    cuda_memory_ptrs_.pop();



    OperationParameters op;

    DataSize3 data_size = {image.Width(), image.Height(), 0};

    CopyData2DtoDevice(image, dev_image, dev_container_size_.height, dev_container_size_.pitch);
    
    op.Clear();
    op.PushValuePtr("image", &dev_image);
    op.PushValuePtr("data_size", &data_size);
    op.PushValuePtr("flow_x", &dev_flow_x);
    op.PushValuePtr("flow_y", &dev_flow_y);
    op.PushValuePtr("corr", &dev_corr);

    op.PushValuePtr("corr_window_size", &correlation_window_size_);

    size_t extended_pitch_size = dev_container_extended_size_.pitch / sizeof(float);
    op.PushValuePtr("pitch_ext", &extended_pitch_size);

    op.PushValuePtr("corr_ext", &dev_container_extended_corr);
    op.PushValuePtr("corr_max_ext", &dev_container_extended_corr_max);

    cuop_correlation_.Execute(op);


    /* Read output data back from the device */
    CopyData2DFromDevice(dev_image, image, dev_container_size_.height, dev_container_size_.pitch);
    CopyData2DFromDevice(dev_flow_x, flow_x, dev_container_size_.height, dev_container_size_.pitch);
    CopyData2DFromDevice(dev_flow_y, flow_y, dev_container_size_.height, dev_container_size_.pitch);
    CopyData2DFromDevice(dev_corr, corr, dev_container_size_.height, dev_container_size_.pitch);

    CopyData2DFromDevice(dev_container_extended_corr, corr_temp, dev_container_extended_size_.height, dev_container_extended_size_.pitch);

    /* Estimate GPU computation time */
    CheckCudaError(cuEventRecord(cu_event_stop, NULL));
    CheckCudaError(cuEventSynchronize(cu_event_stop));

    float elapsed_time;
    CheckCudaError(cuEventElapsedTime(&elapsed_time, cu_event_start, cu_event_stop));

    std::printf("Total GPU computation time: % 4.4fs\n", elapsed_time / 1000.);

    CheckCudaError(cuEventDestroy(cu_event_start));
    CheckCudaError(cuEventDestroy(cu_event_stop));

    /* Return all memory pointers to stack */
    cuda_memory_ptrs_.push(dev_image);
    cuda_memory_ptrs_.push(dev_flow_x);
    cuda_memory_ptrs_.push(dev_flow_y);
    cuda_memory_ptrs_.push(dev_corr);


}

void CorrelationFlow2D::Destroy()
{
    for (CudaOperationBase* cuop : cuda_operations_) {
        cuop->Destroy();
    }

    size_t freed_containers_count = 0;
    while (!cuda_memory_ptrs_.empty()) {
        CUdeviceptr dev_container = cuda_memory_ptrs_.top();
        CheckCudaError(cuMemFree(dev_container));

        ++freed_containers_count;
        cuda_memory_ptrs_.pop();
    }
    if (freed_containers_count && freed_containers_count != dev_containers_count_) {
        std::printf("Warning. Not all device memory allocations were freed.\n");
    }

    /*cout<<"Device containers (before delete):"<<endl;
    cout<<dev_container_extended_corr<<endl;
    cout<<dev_container_extended_corr_max<<endl;

    CheckCudaError(cuMemFree(dev_container_extended_corr));
    CheckCudaError(cuMemFree(dev_container_extended_corr_max));*/


    initialized_ = false;
}



CorrelationFlow2D::~CorrelationFlow2D()
{
    Destroy();
}
