#!/usr/bin/env bash

python run_experiment.py halfmoon wb
python run_experiment.py halfmoon wb_kernel
python run_experiment.py halfmoon kernel
python run_experiment.py halfmoon nn
python plotting_halfmoon.py
