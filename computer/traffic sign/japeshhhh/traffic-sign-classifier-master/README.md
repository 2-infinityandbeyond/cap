## Building a Traffic Sign Recognition Classifier (Self-Driving Car Engineer Nanodegree)

This repository contains the code in a form of a Jupyter notebook that could be
used in trafic signs classification task. The preprocessed data is also presented
in this repository in [`traffic-sign-data`](./traffic-sign-data). The data
contains German traffic sign images of 43 kinds 
([description](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)).
The model of convolutional neural network that performs classification
can be found in [`Traffic_Sign_Classifier.ipynb`](./Traffic_Sign_Classifier.ipynb). Please, follow
the instructions below to prepare an environment to run this notebook.
There is also [`Writeup.md`](./Writeup.md) where the pros and cons of the implementation described.

#### Preparing Environment
In order to be able to use traffic sign data in this repository, Git Large File Storage extension 
should be installed on the system. Installation instructions can be found on 
[git-lfs.github.com](https://git-lfs.github.com).

To run Jupyter notebook, start a docker container with:
```bash
docker run --interactive --tty --rm --publish 8888:8888 --volume $PWD:/src udacity/carnd-term1-starter-kit:latest
```
Note, that the current directory will become your working directory in Jupyter notebook.

Then, copy a link from the console to your browser and start exploring
the source code. The link will be similar but not equal to
`http://localhost:8888/?token=eb26e4a2b935c384dc3e0230a8181984f07da6be9df0c1b8`.

#### Notice
Some functions from [`Traffic_Sign_Classifier.ipynb`](./Traffic_Sign_Classifier.ipynb) are
provided by [Udacity.com](https://www.udacity.com).

