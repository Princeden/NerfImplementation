### Simple NeRF Implementation

Implementation of the NERF model using PyTorch and then integrated with the PYCOLMAP library.

#### Explanation

The [original paper](https://www.matthewtancik.com/nerf) does a very good job explaining the thought process behind the model. The program will automatically use GPU if it is possible. It is important to note that COLMAP requires incredibly high amounts of RAM and without a GPU it is likely that it will be killed by your kernel. 


