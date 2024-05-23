# Code for paper "Differentiable Distributionally Robust Optimization Layers"

This repository contains the source code to reproduce the experiments of our paper "Differentiable Distributionally Robust Optimization Layers" presented in ICML 2024.

## Dependency

Except for commonly used packages like torch and numpy, other important packages our code is built on are listed as follows.

- cvxpylayers
- cvxpy
- gurobipy
  
Moreover, make sure that [Gurobi](https://www.gurobi.com/) solver can be called in Python environment. This [tutorior](https://support.gurobi.com/hc/en-us/articles/360044290292-How-do-I-install-Gurobi-for-Python) may be helpful for building Gurobi in Python.

## Usage

Code for experiments on the multi-item newsvendor problem is in folder 'Toy_Example_Newsvendor', and running the main file in this folder will conduct this experiment. 

Code for experiments on the portfolio management problem is in folder 'Portfolio_Management_Problem'. Under this folder, the 'Pure_Continous_Decision_Case' folder contains experiment when all decision variables are continuous and 'Mixed_Integer_Decision_Case' folder contains experiment when decision variables are mixed-integer. Simply running the main file in folders 'Portfolio_Management_Problem/Pure_Continous_Decision_Case' and 'Portfolio_Management_Problem/Mixed_Integer_Decision_Case' will conduct corresponding experiments.

For anyone who wants to construct a DRO Layer for their own problems, using the [cvxpylayers](https://github.com/cvxgrp/cvxpylayers) package directly when all decision variables are continuous. When the decision variables are mixed-integer, we provide MIDROLayer class (in 'Portfolio_Management_Problem\Mixed_Integer_Decision_Case\my_layer_MI.py') for computing and back-propagating the gradient.  


