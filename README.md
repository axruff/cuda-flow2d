# GPU-based 2D Optical flow using NVIDIA CUDA

Optical flow using *variational methods* which determine the unknown displacement field as a minimal solution
of the energy functional. 

In general, such energy-based formulations are composed of two
parts: a *data term* which assumes constancy of specific image features, and a *smoothness term* which regularizes the spatial variation of the flow field.

## Features

* **Computational model guarantees**:
   * a unique solution (global optimal solution)
   * stability of the algorithm (no critical dependence on parameters)
* **High quality**:
   * dense flow results (one displacement for each pixel)
   * allows large displacements
   * different types of motion (translation, rotation, local elastic transformation)
   * sub-pixel accuracy
* **Robustness**:
   * under noise
   * under varying illumination (brightness changes) 
   * with respect to artifacts

## Example

![alt text](https://github.com/axruff/cuda-flow2d/raw/master/examples/insect.png "Moving insect")

**Figure**: Head of the feeding cockroach *Periplaneta americana* imaged by fast X-ray radiography. **(a)** First frame of the radiographic sequence. **(b)** Computed flow field, which captures the movements of the insect during chewing process. **(c)** Color coding: color represents direction and its brightness represents flow magnitude.

## Model


In a short form The 2D variational optical flow model using *grey value constancy* and the *flow-driven smoothness* approach is given by:

<img src="https://render.githubusercontent.com/render/math?math=e^{i \pi} = -1">

<a href="https://www.codecogs.com/eqnedit.php?latex=$$&space;E_{2D}(u,v)&space;=&space;\int_{\Omega_{2}}{|I(x&plus;u,y&plus;v,t&plus;1)&space;-&space;I(x,y,t)|&space;&plus;&space;\alpha&space;\Psi(|\nabla_{2}u|^2&space;&plus;&space;|\nabla_{2}v|^2)&space;\text{d}x&space;\text{d}y}.&space;$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$&space;E_{2D}(u,v)&space;=&space;\int_{\Omega_{2}}{|I(x&plus;u,y&plus;v,t&plus;1)&space;-&space;I(x,y,t)|&space;&plus;&space;\alpha&space;\Psi(|\nabla_{2}u|^2&space;&plus;&space;|\nabla_{2}v|^2)&space;\text{d}x&space;\text{d}y}.&space;$$" title="$$ E_{2D}(u,v) = \int_{\Omega_{2}}{|I(x+u,y+v,t+1) - I(x,y,t)| + \alpha \Psi(|\nabla_{2}u|^2 + |\nabla_{2}v|^2) \text{d}x \text{d}y}. $$" /></a>

I(x, y, t) is a sequence of input images, where (x, y) are the pixel coordinates within a rectangular image domain
and t denotes time frame: u, v - are the unknown displacement components and 

<a href="https://www.codecogs.com/eqnedit.php?latex=$\Psi(x)&space;=&space;\sqrt{x&space;&plus;&space;\epsilon}$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$\Psi(x)&space;=&space;\sqrt{x&space;&plus;&space;\epsilon}$" title="$\Psi(x) = \sqrt{x + \epsilon}$" /></a>

is the smoothness penalty function which corresponds to the total variation (TV) regularization.


**Features of the optical flow model**:
* Brightness constancy data term [1]
* Gradient constancy data term [2]
* Data term based on higher-order derivatives [3]
* Robust modeling of data term [4]
* Flow-driven smoothness [3]
* Coarse-to-fine flow estimation [2]
* Intermediate flow median filtering [5]
 
 
 ## References
 
* [1] B. Horn and B. Schunck. *Determining optical flow*. Artificial Intelligence, 17:185{203, 1981.
* [2] T. Brox, A. Bruhn, N. Papenberg, and J. Weickert. *High accuracy optic flow estimation based on a theory for warping*. In T. Pajdla and J. Matas, editors, Computer Vision, ECCV 2004, volume 3024 of Lecture Notes in Computer Science, pages 25-36. Springer, Berlin, 2004.
* [3] N. Papenberg, A. Bruhn, T. Brox, S. Didas, and J. Weickert. *Highly accurate optic ow computation with theoretically justified warping*. Int. J. Comput. Vision, 67(2):141-158, April 2006.
* [4] M. J. Black and P. Anandan. *The robust estimation of multiple motions: Parametric and piecewise-smooth flow fields*. Computer Vision and Image Understanding, 63(1):75- 104, 1996.
* [5] D. Sun, S. Roth, and M. J. Black. *A quantitative analysis of current practices in optical flow estimation and the principles behind them*. International Journal of Computer Vision, 106(2):115-137, 2014.
