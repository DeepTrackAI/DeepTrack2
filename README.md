# DeepTrack 
By Saga Helgadottir, Aykut Argun and Giovanni Volpe.

DeepTrack is a trainable convolutional neural network that predicts the positon of objects in microscope images. This is the code for the paper [Digital video microscopy enhanced by deep learning](https://arxiv.org/abs/1812.02653 "Digital video microscopy enhanced by deep learning"). 

<img src="https://github.com/giovannivolpe/DeepTrack/blob/develop/figures/DeepTrack architecture.png" />

## Dependencies 
* Python 3.6 
* Keras (v 2.2.4 or higher)
* Tensorflow 
* Pillow
* Opencv
* Pandas

## Usage
Each code example is a Jupyter Notebook that also includes detailed comments to guide the user. All neccesary files to run the code examples are provided. 

The network is trained on various kinds of simulated images of particles with given ground truth positions, optimized for each problem. The particles are represented by combinations of Bessel functions and their size, shape and intensity can be changed. In addition, the image background level, signal-to-noise level and illumination gradient can be changed. A few examples are shown below:

<img src="https://github.com/giovannivolpe/DeepTrack/blob/develop/figures/sample_image_10.png" width="150" height="150" /> <img src="https://github.com/giovannivolpe/DeepTrack/blob/develop/figures/sample_image_9.png" width="150" height="150" /> <img src="https://github.com/giovannivolpe/DeepTrack/blob/develop/figures/sample_image_8.png" width="150" height="150" /> <img src="https://github.com/giovannivolpe/DeepTrack/blob/develop/figures/sample_image_7.png" width="150" height="150" /> <img src="https://github.com/giovannivolpe/DeepTrack/blob/develop/figures/sample_image_6.png" width="150" height="150" /> <img src="https://github.com/giovannivolpe/DeepTrack/blob/develop/figures/sample_image_5.png" width="150" height="150" /> <img src="https://github.com/giovannivolpe/DeepTrack/blob/develop/figures/sample_image_4.png" width="150" height="150" /> <img src="https://github.com/giovannivolpe/DeepTrack/blob/develop/figures/sample_image_3.png" width="150" height="150" /> <img src="https://github.com/giovannivolpe/DeepTrack/blob/develop/figures/sample_image_2.png" width="150" height="150" /> <img src="https://github.com/giovannivolpe/DeepTrack/blob/develop/figures/sample_image_1.png" width="150" height="150" /> 

After the network has been trained it can be used to track different kind of objects in images. For example, the particles and bacteria in the video below can be tracked seperately:

![](https://github.com/giovannivolpe/DeepTrack/blob/develop/figures/sample_video.gif)


## Citations

DeepTrack is an open-source library and is licensed under the GNU General Public License (v3). For questions contact Giovanni Volpe at giovanni.volpe@physics.gu.se. If you are using this library please cite:

Saga Helgadottir, Aykut Argun, and Giovanni Volpe. "Digital video microscopy enhanced by deep learning." Optica 6.4 (2019): 506-513.



## Funding
This work was supported by the ERC Starting Grant ComplexSwimmers (Grant No. 677511).

