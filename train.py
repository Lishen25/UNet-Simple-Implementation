# %%
# USAGE
# python train.py
# import the necessary packages
from pyimagesearch.dataset import SegmentationDataset
from pyimagesearch.model import UNet
from pyimagesearch import config
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms

from imutils import paths # --> THIS PACKAGE IS TO MAKE OPENCV OPERATION MORE CONVENIENT

from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import time
import os
import torch.nn as nn

# load the image and mask filepaths in a sorted manner
imagePaths = sorted(list(paths.list_images(config.TRAIN_IMAGE_DATASET_PATH)))
maskPaths = sorted(list(paths.list_images(config.TRAIN_MASK_DATASET_PATH)))
# THESE TWO PATHS ARE LISTS

# %%
# partition the data into training and testing splits using 85% of the data for training and the remaining 15% for testing
trainImages, valImages, trainMasks, valMasks = train_test_split(imagePaths, maskPaths, test_size=config.VAL_SPLIT, random_state=42)
# THIS IS THE SAME AS THE CODE BELOW
# trainImages, testImages = train_test_split(imagePaths, test_size=config.VAL_SPLIT, random_state=42)
# trainMasks, testMasks = train_test_split(maskPaths, test_size=config.VAL_SPLIT, random_state=42)

# write the testing image paths to disk so that we can use then
# when evaluating/testing our model
print("[INFO] saving validation image paths...")
f = open(config.VAL_PATHS, "w")
f.write("\n".join(valImages))
f.close()

# %%
# define transformations
ImageTransforms = transforms.Compose([transforms.ToPILImage(), 
                                 transforms.Resize((config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH)), 
                                 transforms.ToTensor()])

MaskTransforms = transforms.Compose([transforms.ToPILImage(), 
                                 transforms.Resize((config.OUT_IMAGE_HEIGHT, config.OUT_IMAGE_WIDTH)), 
                                 transforms.ToTensor()])

# create the train and test datasets
trainDS = SegmentationDataset(imagePaths=trainImages, maskPaths=trainMasks, 
							  ImageTransforms=ImageTransforms, MaskTransforms=MaskTransforms)
valDS = SegmentationDataset(imagePaths=trainImages, maskPaths=trainMasks, 
							  ImageTransforms=ImageTransforms, MaskTransforms=MaskTransforms)

print(f"[INFO] found {len(trainDS)} examples in the training set...")
print(f"[INFO] found {len(valDS)} examples in the test set...")

# create the training and test data loaders
trainLoader = DataLoader(trainDS, shuffle=True, batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY, num_workers=os.cpu_count())
valLoader = DataLoader(valDS, shuffle=False, batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY, num_workers=os.cpu_count())
# The num_workers here refers to the threads in the CPU, which is related to cpu

# initialize our UNet model
unet = UNet(outSize=(config.OUT_IMAGE_HEIGHT, config.OUT_IMAGE_WIDTH)).to(config.DEVICE)

# initialize loss function and optimizer
lossFunc = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(unet.parameters(), lr=config.INIT_LR)

# calculate steps per epoch for training and test set
trainSteps = len(trainDS) // config.BATCH_SIZE
valSteps = len(valDS) // config.BATCH_SIZE
# These two steps are in fact batch numbers

# initialize a dictionary to store training history
H = {"train_loss": [], "validation_loss": []}

# To save the best model
best_valid_loss = config.BEST_VALID_LOSS

# %%
# loop over epochs
print("[INFO] training the network...")
startTime = time.time()
for e in tqdm(range(config.NUM_EPOCHS)):
	# set the model in training mode
	unet.train()
	
	# initialize the total training and validation loss
	totalTrainLoss = 0
	totalValLoss = 0
	
	# loop over the training set
	for x, y in trainLoader:
		# send the input to the device
		x = x.to(config.DEVICE)
		y = y.to(config.DEVICE)
        
		# perform a forward pass and calculate the training loss
		pred = unet(x)
		
		loss = lossFunc(pred, y)
		
		# first, zero out any previously accumulated gradients, then
		# perform backpropagation, and then update model parameters
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		
		# add the loss to the total training loss so far
		totalTrainLoss += loss
		
	# switch off autograd
	with torch.no_grad():
		# set the model in evaluation mode
		unet.eval()
		
		# loop over the validation set
		for x, y in valLoader:
			# send the input to the device
			x = x.to(config.DEVICE)
			y = y.to(config.DEVICE)
			# make the predictions and calculate the validation loss
			pred = unet(x)
			totalValLoss += lossFunc(pred, y)
			
	# calculate the average training and validation loss
	avgTrainLoss = totalTrainLoss / trainSteps
	avgValLoss = totalValLoss / valSteps
	
	# update our training history
	H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
	H["validation_loss"].append(avgValLoss.cpu().detach().numpy())
	# .cpu() means move the variable to cpu
	# .detach() means detach the variable from computation graph
	# .numpy() means transfer the variable into numpy form
	
	# print the model training and validation information
	print("[INFO] EPOCH: {}/{}".format(e + 1, config.NUM_EPOCHS))
	print("Train loss: {:.6f}, Validation loss: {:.4f}".format(avgTrainLoss, avgValLoss))
	
    # Save the model if its the best
	if avgValLoss < best_valid_loss:
		best_valid_loss = avgValLoss
		torch.save(unet.state_dict(), config.KAGGLE_MODEL_PATH)
	
# display the total time needed to perform the training
endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))

# %%
# plot the training loss
plt.style.use("ggplot")
plt.figure()
plt.plot(H["train_loss"], label="train_loss")
plt.plot(H["validation_loss"], label="validation_loss")
plt.title("Training Loss on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig(config.PLOT_PATH)