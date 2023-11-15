# VirtualMultiplexing3D
## Overview
VM-3D is a 3D virtual multiplexing model. Build upon the Vox2Vox volume-to-volume translation model.

## What type of data does VM-3D work with?
- Current pretrained model works with mixed signals from KI67 and CDH1
## What output can RDC-org provide?
- Unmixed signals from two or more mixed signals
## Software and Hardware requirements
VM-3D runs with Python 3.9.0.

The VM-3D virtual multiplexing pipeline can be run on any decent computer with CUDA capable graphics hardware.

For image analysis we made use of a workstation with the following specs:
| | |
| ------------- | ------------- |
| GPU |		NVIDIA Quadro RTX 6000 |
| Processor | **2**	Intel(R) Xeon(R) Gold 5122 CPU @ 3.60GHz  |
| RAM |	200 GB |
|System | type	64-bit operating system, x64-based processor |


## Installation
Download the repository to your PC via direct dowload or git clone https://github.com/AlievaRios/VirtualMultiplexing.git in Git Bash.\
Installation should take <15 minutes, but due to Conda issues it could take some time.

VM-3D uses the STAPL3D library (version used with Python 3.9.0) :
Install using their instructions: https://github.com/RiosGroup/STAPL3D
or install all packages from the package-list.txt using Conda 
```
conda create -n myenv --file package-list.txt
```

## Input data
The current version of the VM-3D model has been trained and tested with the mixed channels from KI67 and CDH1 in 3D fluorescence scans of organoids.

## Repository
This repository contains a collection of scripts enabling the following dowstream analysis.

## Set-up

## Use pre-trained models

One pre-trained model can be downloaded from this [drive link](https://drive.google.com/file/d/1lxCriTK1MnWaDFuEir01BsZMumcmL5SF/view?usp=share_link). This was tested and trained using 3D fluorescence imaging with the mixed channels from KI67 and CDH1.
The input data expected by the VM-3D model for training and prediction are h5 files, with a dataset with the name "image".


## Demo

For a demo, check the jupiter notebook, VM-vox2vox_example.ipynb

To use config files with the command line, some example configs are included in the configs folder.

### Start
Set your directory to the project folder
```
cd <Project folder>
```

### Preprocessing Training Dataset
The preprocessor expects a czi file.
```
python ./preprocessing/Preprocessor.py -c ./configs/1.Preprocessing_Training.yml
```

### Preprocess Testing Dataset
The preprocessor expects a czi file.
```
python ./preprocessing/Preprocessor.py -c ./configs/2.Preprocessing_Testing.yml
```


### Training a model
```
python ./train.py -c ./configs/3.Training.yml
```

### Predict testing file
```
python ./predict.py -c ./configs/4.predict_conf.yml
```

### Combine h5 files
```
python ./postprocessing/postprocessor.py -c ./configs/5.combine.conf.yml
```

### Scoring results
```
python ./test.py -c ./configs/6.scoring.yml
```

### Convert Files to Pix2Pix format for Testing
```
python ./preprocessing/Converter.py -c ./configs/7.convert_to_pix2pix.yml
```

### Convert Files from Pix2Pix for Testing
Make sure the folder used for the earlier conversion contains the folder with the pix2pix results in the directory of "pix2pix_results"
```
python ./preprocessing/Converter.py -c ./configs/8.convert_from_pix2pix.yml
```


