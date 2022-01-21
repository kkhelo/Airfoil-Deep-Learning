Aerodynamic Coefficient Prediction Deep Learning
===============

This repo is built for predict aerodynamics coefficient like cl, cd by using deep learning technique.

Table of Content
-----------------
* Background
* Demo Install
* Usage
* Maintainers

Background
----------

CFD is the technique that allowing us to calculate velocity field, pressure field etc. around any objects. <br>
But the computation time is always the main issue for researchers. <br>
Our main goal is to train a model that can help CFD decrease time costs, or even replace it.

Demo Install
-----------

Comming soon!

Usage
------------

Before training, take a look at `local dataset helper` folder :
* `set_builder.py` helps you make dataset that can be feed into our custom torch.Dataset.<br>  
  It can also expand airfoil df plot to 3 channels, by using the gradient of df plot.<br><br>
* Others :  Comming soon.<br>

After you build the dataset from `set_builder.py`, you can use `train.py` to train the model 

Maintainers
---------
@kkhelo <br>

