#!/bin/bash

nohup python Autoencoder/Socket64Dense.py $1 &
cd ./ConvAE
matlab -nodesktop -nosplash -r "StartSocket($1)"
cd ../../
