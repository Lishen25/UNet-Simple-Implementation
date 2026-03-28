THIS PROJECT COMES FROM THE ONLINE TUTORIAL: https://pyimagesearch.com/2021/11/08/u-net-training-image-segmentation-models-in-pytorch/

# THE STRUCTURE OF THIS PROJECT
.
├── dataset
│   └── train
├── output
├── predict.py
├── pyimagesearch
│   ├── config.py
│   ├── dataset.py
│   └── model.py
└── train.py

# FUNCTION OF EACH DIRECTORY
dataset: Stores the dataset for the model
    train: training dataset

output: Stores the trained model and training loss plots

pyimagesearch: 
    config.py: Stores the parameters of the code, initial settings and the configuration
    dataset.py: Stores the definition of the Dataset class(defined by ourselves)
    model.py: Stores the structure/definition of the model

train.py: training codes

prediction.py: prediction codes