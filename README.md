# DeepTrack 
By Saga Helgadottir, Aykut Argun and Giovanni Volpe.

DeepTrack is a trainable convolutional neural network that predicts the positon of objects in microscope images. This is the code for the paper [Digital video microscopy enhanced by deep learning](https://arxiv.org/abs/1812.02653 "Digital video microscopy enhanced by deep learning"). Each code example is a Jupyter Notebook that also includes detailed comments to guide the user. All neccesary files to run the code examples are provided. 

The network is trained on various kinds of simulated images with given ground truth positions, optimized for each problem (**add different images**): 

<img src="https://github.com/giovannivolpe/DeepTrack/blob/develop/ex2_24.png" width="150" height="150" /> <img src="https://github.com/giovannivolpe/DeepTrack/blob/develop/ex2_5.png" width="150" height="150" /> <img src="https://github.com/giovannivolpe/DeepTrack/blob/develop/ex2_7.png" width="150" height="150" />

After the network has been trained it can be used to track different kind of objects in images:

![](video4.gif)


## Dependencies 
* Python 3.6 
* Keras (v 2.2.4 or higher)
* Tensorflow 
* Pillow
* Opencv
* Pandas


## Future work 
* Training DeepTrack to detect objects of irregular shapes
* Training DeepTrack to detect the orientation of objects
* Please report any issues in the issue section so that the code can be updated accordingly 

## Citations

DeepTrack is an open-source library and is licensed under the GNU General Public License (v3)???. For questions contact Giovanni Volpe at giovanni.volpe@physics.gu.se. If you are using this library please cite:

Saga Helgadottir, Aykut Argun, and Giovanni Volpe. "Digital video microscopy enhanced by deep learning." arXiv preprint arXiv:1812.02653 (2018).
