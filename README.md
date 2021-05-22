# Deep Learning with MNIST
This file can be run locally or on a cloud. When using the Pittsburgh Supercomputing Center (PSC), first connect to a GPU node from the base node using `interact -gpu`, and then load the tensorflow2 container (The commend I used was `singularity shell --nv /ocean/containers/ngc/tensorflow/tensorflow_20.02-tf2-py3.sif`). Finally, run python (`python`) and `import tensorflow`.

## Overview
This is a take-home exercise from the XSEDE Big Data and Machine Learning Workshop.

I worked with the built-in MNIST dataset. I built a `Sequential` model, which includes 2 `Conv2D` layers to get 2D feature maps, a `MaxPooling2D` layer to reduce dimension and reduce overfitting, and 2 1D `Dense` layers. The model also includes 2 `Dropout` layers to reduce overfitting. Finally, I plotted the training and test scores to evluate of the model performed.

## TensorBoard
There is a line of code that allows for monitoring the training of the model in real time using TensorBoard that starts with `tensorboard_callback = ...`. This line, along with specifying `callbacks = [tensorboard_callback]` in `model.fit()` create a folder (named `TB_logDir` in my code) in the current directory and can be read by TensorBoard. 

To open Tensorboard, run commend `tensorboard --logdir TB_logDir` in terminal. Note that the directory should point to the folder created by the code mentioned above. After running the commend, you should see a prompt like `TensorBoard 2.4.0 at http://localhost:6006/ (Press CTRL+C to quit)`. Enter the url in a web browser to use Tensorboard. You should be able to see plots of training and test scores, among other things.

![Alt text](TensorBoard.png?raw=true "Title")
