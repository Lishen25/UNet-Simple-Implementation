# import the necessary packages
from torch.utils.data import Dataset
import cv2

class SegmentationDataset(Dataset):
	def __init__(self, imagePaths, maskPaths, ImageTransforms, MaskTransforms):
		# store the image and mask filepaths, and augmentation
		# transforms
		self.imagePaths = imagePaths
		# imagePaths is a list consists of path of images

		self.maskPaths = maskPaths
		self.ImageTransforms = ImageTransforms
		self.MaskTransforms = MaskTransforms
		
	def __len__(self):
		# return the number of total samples contained in the dataset
		return len(self.imagePaths)
	
	def __getitem__(self, idx):
		# grab the image path from the current index
		imagePath = self.imagePaths[idx]
		
		# load the image from disk, swap its channels from BGR to RGB,
		# and read the associated mask from disk in grayscale mode
		image = cv2.imread(imagePath)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # opencv read images as BGR order, but RGB is used more ofter
		# so change it to RGB manually

		mask = cv2.imread(self.maskPaths[idx], 0)
		# 0 means change this image to grayscale picture automatically

		# check to see if we are applying any transformations
		# apply the transformations to both image and its mask
		if self.ImageTransforms is not None:
			image = self.ImageTransforms(image)
			
		if self.MaskTransforms is not None:
			mask = self.MaskTransforms(mask)
		
		# return a tuple of the image and its mask
		return (image, mask)
	
class TestDataset(Dataset):
	def __init__(self, imagePaths, transforms):
		self.imagePaths = imagePaths
		self.transforms = transforms
		
	def __len__(self):
		return len(self.imagePaths)
	
	def __getitem__(self, idx):
		imagePath = self.imagePaths[idx]
		image = cv2.imread(imagePath)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		
		if self.transforms is not None:
			# apply the transformations to both image and its mask
			image = self.transforms(image)

		return image