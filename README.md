# Optimal networks

For the paper "A convolutional neural network reaches optimal sensitivity for detecting some but not all patterns".  
Reith and Wandell. (arXiv)

Abstract
We investigate the spatial contrast sensitivity of modern convolutional neural networks (CNNs) and a linear support vector machine (SVM). To measure performance, we compare the CNN contrast-sensitivity across a range of patterns with the contrast-sensitivity of a Bayesian ideal observer (IO) with the signal-known-exactly and noise-known statistically. A ResNet-18 reaches optimal performance for harmonic patterns, as well as several classes of real world signals including faces. For these stimuli the CNN substantially outperforms the SVM. We further analyze the case in which the signal might appear in one of multiple locations and found that CNN spatial sensitivity continues to match the IO. However, the CNN sensitivity is far below optimal at detecting certain complex texture patterns. These measurements show that CNNs contrast-sensitivity differs markedly between spatial patterns. The variation in spatial contrast sensitivity may be a significant factor, influencing the performance level of an imaging system designed to detect low contrast spatial patterns.

The signal generation code is in Matlab and depends on ISETCam (https://github.com/iset/isetcam/wiki).  

The network training code is in Python and imports various libraries.

* The script <> is an overview of the whole computational stream. 
* Script <> generates the data points in Figure 2
* Script <> generates the data points in Figure 3






