Introduction
============
This library includes the experiments used for the development of the
algorithm as described in the paper

    V. Dhiman, J. Ryde, and J. J. Corso. Mutual localization: Two camera relative 6-dof pose estimation from reciprocal fiducial observation. In Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), November 2013.

A copy of paper is supplied with this release: `download paper`_. 

.. _download paper: _static/paper.pdf

This document intends to relate code with the paper.

Installation
============

To run the experiments, first we need to install the python dependencies that
are required for the code. Some dependencies are essential for the core of the
algorithm other are required for some peripheral code. 

Dependencies
------------

    - Opencv : cv2
    - Numpy : numpy
    - matplotlib
    - mayavi2
    - scipy.linalg
    - scipy.ndimage
    - scipy.optimize
    - logging
    - pyexiv2
    - pygame
    - sympy
    - yaml
    - unittest
    - texlive

Installing core dependencies::

    sudo apt-get install libopencv python-opencv python-numpy python-sympy python-matplotlib python-scipy python-yaml mayavi2 texlive

Installing other dependencies::

    sudo apt-get install python-unittest python-pyexiv2 python-pygame

Download
========

Download from `here`_.::

    git clone git@github.com:wecacuee/mutual_localization.git

Get data by using the script in data directory:
    
    cd data/
    python wgetdata.py

Setting python path::

    cd mutual_localization/
    export PYTHONPATH=$PYTHONPATH:`pwd`/lib:`pwd`/src

For the rest of the documentation we will assume that we will assume that we
are in the ``mutual_localization`` directory.

.. _here: https://github.com/wecacuee/mutual_localization
