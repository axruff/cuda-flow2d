# GPU-based 2D Optical flow using NVIDIA CUDA

Optical flow using variational methods which determine the unknown displacement field as a minimal solution
of the energy functional. 

In general, such energy-based formulations are composed of two
parts: a data term which assumes constancy of specific image features, and a smoothness
term which regularizes the spatial variation of the flow field.

## Features

* Computational model guarantees:
   * a unique solution (global optimal solution)
   * stability of the algorithm (no critical dependence on parameters)
* High quality:
   * dense flow results (one displacement for each pixel)
   * allows large displacements
   * different types of motion (translation, rotation, local elastic transformation)
   * sub-pixel accuracy
* Robustness:
   * under noise
   * under varying illumination (brightness changes) 
   * with respect to artifacts

## Example

![alt text](https://github.com/axruff/cuda-flow2d/raw/master/examples/insect.png "Moving insect")
Figure: Fast radiography of the feeding cockroach Periplaneta americana. (a) First frame of the radiographic sequence (the background is removed and contrast is adjusted). (b) Computed flow field, which captures the movements of the insect. (c) Color coding: color represents direction and its brightness represents flow magnitude.

## Model

