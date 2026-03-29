# ORIGIN AND THE GOAL OF THIS PROJECT
THE MAJORITY OF THIS PROJECT COMES FROM THE ONLINE TUTORIAL: https://pyimagesearch.com/2021/11/08/u-net-training-image-segmentation-models-in-pytorch/

This project may provide a simple example and implementation of how to construct
a UNet for a given dataset

# THE STRUCTURE OF THIS PROJECT
```text
.
├── dataset
│   ├── test
│   └── train
├── output
├── prediction.py
├── pyimagesearch
│   ├── config.py
│   ├── dataset.py
│   └── model.py
├── show_val_result.py
└── train.py
```

# FUNCTION OF EACH DIRECTORY
dataset: Stores the dataset for the model

train: training dataset

output: Stores the trained model and training loss plots

pyimagesearch: \
config.py: Stores the parameters of the code, initial settings and the configuration\
dataset.py: Stores the definition of the Dataset class(defined by ourselves)\
model.py: Stores the structure/definition of the model\

train.py: training codes

show_val_result: illustrate the difference between the result of validation and the ground-truth mask

prediction.py: prediction codes