# Pneumothorax Algorithm

This is a Docker container for the pneumothorax algorithm from: https://github.com/i-pan/kaggle-siim-ptx/

Note that this container is built to use a 3-fold ensemble of segmentation models with horizontal flip test time augmentation. Test DICOMs are available in `./samples`. These samples were taken from the Kaggle SIIM-ACR Pneumothorax Segmentation competition training set and are for visualization purposes only. 

## Setup Instructions

Make sure you have Docker installed. 

Download model checkpoints:
```
pip install gdown
cd kaggle-siim-ptx
gdown https://drive.google.com/uc?id=1lGI-SScvF2SBnwakqjl13RqpVM5R-lyD
unzip checkpoints.zip ; rm checkpoints.zip
cd ..
```

Setup container:
```
docker build -t ipan-ptx:0.0 .
```

Run container:
```
docker run --volume=$(pwd):/io ipan-ptx:0.0 samples/dicom_000
```
