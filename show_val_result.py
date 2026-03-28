# %%
# USAGE
# import the necessary packages
from pyimagesearch import config
from pyimagesearch import model
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import os

# THIS FUNCTION IS FOR ILLUSTRATING THE RESULT
def prepare_plot(origImage, origMask, predMask):
	# initialize our figure
	figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))
	
	# plot the original image, its mask, and the predicted mask
	ax[0].imshow(origImage)
	ax[1].imshow(origMask)
	ax[2].imshow(predMask)
	
	# set the titles of the subplots
	ax[0].set_title("Image")
	ax[1].set_title("Original Mask")
	ax[2].set_title("Predicted Mask")
	
	# set the layout of the figure and display it
	figure.tight_layout()
	figure.show()

# THIS IS THE PREDICTION LOOPS
def make_predictions(model, imagePath):
	# set model to evaluation mode
	model.eval()
	
	# turn off gradient tracking
	with torch.no_grad():
		# load the image from disk, swap its color channels, cast it
		# to float data type, and scale its pixel values
		image = cv2.imread(imagePath)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = image.astype("float32") / 255.0
		
		# resize the image and make a copy of it for visualization
		image = cv2.resize(image, (128, 128))
		orig = image.copy()
		
		# find the filename and generate the path to ground truth mask
		filename = imagePath.split(os.path.sep)[-1]
		groundTruthPath = os.path.join(config.TRAIN_MASK_DATASET_PATH, filename)
		
		# load the ground-truth segmentation mask in grayscale mode
		# and resize it
		gtMask = cv2.imread(groundTruthPath, 0)
		gtMask = cv2.resize(gtMask, (config.OUT_IMAGE_WIDTH, config.OUT_IMAGE_HEIGHT))
		
        # make the channel axis to be the leading one, add a batch
		# dimension, create a PyTorch tensor, and flash it to the current device
		image = np.transpose(image, (2, 0, 1))              # reorder the dimension of the image
		image = np.expand_dims(image, axis=0)               # axis=0 means add a new dimension in the 1st place and the corresponding number is 1
		image = torch.from_numpy(image).to(config.DEVICE)   # change it to a tensor
		
		# make the prediction, pass the results through the sigmoid
		# function, and convert the result to a NumPy array
		predMask = model(image).squeeze()
		predMask = torch.sigmoid(predMask)
		predMask = predMask.cpu().numpy()
		
		# filter out the weak predictions and convert them to integers
		predMask = (predMask > config.THRESHOLD) * 255
		predMask = predMask.astype(np.uint8)

		# prepare a plot for visualization
		prepare_plot(orig, gtMask, predMask)
		
# load the image paths in our testing file and randomly select 10 image paths
print("[INFO] loading up validation image paths...")
imagePaths = open(config.VAL_PATHS).read().strip().split("\n")
imagePaths = np.random.choice(imagePaths, size=10)

# load our model from disk and flash it to the current device
print("[INFO] load up model...")
unet = model.UNet().to(config.DEVICE)
unet.load_state_dict(torch.load(config.KAGGLE_MODEL_PATH, weights_only=True))
unet.eval()

# %%
# iterate over the randomly selected test image paths
for path in imagePaths:
	# make predictions and visualize the results
	make_predictions(unet, path)
# %%
