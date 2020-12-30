# noisystudent-lite
The repo for my final project, "Semi-supervised Learning and Robustness on a Smaller Scale" for [CSCI 2952-C](http://cs.brown.edu/courses/csci2952c/).

The best image recognition models as of 2020 are exceptionally good at classifying images. Since 2010, the premier benchmark for evaluating image classification performance has been [ImageNet-1k](https://paperswithcode.com/sota/image-classification-on-imagenet). Traditionally, researchers would train their models to classify these images from ImageNet via a process called _supervised learning_. Here, the model tries to predict what an image's label is and adjusts itself once it receives the correct label.

Increasingly, the best models have relied on **semi-supervised learning**, a process which leverages hundreds of millions or billions of unlabeled images as extra training data to boost performance. These studies have largely been done by Google Brain or Facebook AI, which have access to storage and computational resources most researchers do not have. Thus, in this study, I examine whether the approach of one of these papers, "Self-training with Noisy Student", can boost a model’s ability to classify images and robustness to noise **even when applied at a smaller scale.**

## Project Structure
All code used to train models are in `src/` folder. Experimental runs and
notebooks are in the `experiments/` and `notebooks/` folders.

Report can be found [here](https://github.com/tyj144/noisystudent-lite/blob/main/Report.pdf).

Credit to the [Brown University Center for Computation and Visualization (CCV)](https://ccv.brown.edu/) for computational resources for this project. Runs from those experiments are saved in the `runs/` folder.
