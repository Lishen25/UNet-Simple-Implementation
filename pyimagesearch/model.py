# import the necessary packages
from . import config
# . means the current file folder(it is called a package in Python)
# Here config refers the config.py in this file folder

from torchvision.transforms import CenterCrop
import torch
import torch.nn as nn

class Block(nn.Module):
	def __init__(self, inChannels, outChannels):
		super(Block, self).__init__()
		# store the convolution and RELU layers
		self.conv1 = nn.Conv2d(inChannels, outChannels, 3)
		self.relu = nn.ReLU()
		self.conv2 = nn.Conv2d(outChannels, outChannels, 3)
		
	def forward(self, x):
		# apply CONV => RELU => CONV block to the inputs and return it
		return self.relu(self.conv2(self.relu(self.conv1(x))))
	
class Encoder(nn.Module):
	def __init__(self, channels=(3, 16, 32, 64)):
		super(Encoder, self).__init__()
		# store the encoder blocks and maxpooling layer
		self.encBlocks = nn.ModuleList([Block(channels[i], channels[i + 1])	for i in range(len(channels) - 1)])
		# Explain using a example here: channels=(3, 16, 32, 64)
        # len(channels) here is 4, thus range(len(channels)-1) == [0,1,2]
		# Block(channels[i], channels[i+1]) creates a new block
		# So here the result in the parentheses after ModuleList is:
		# [Block(3,16), Block(16,32), Block(32,64)]
		# Then they are saved in ModuleList
		
		self.pool = nn.MaxPool2d(2)
		
	def forward(self, x):
		# initialize an empty list to store the intermediate outputs
		blockOutputs = []
		# loop through the encoder blocks
		for block in self.encBlocks:
			# pass the inputs through the current encoder block, store
			# the outputs, and then apply maxpooling on the output
			x = block(x)
			blockOutputs.append(x) # Save the intermediate results
			x = self.pool(x)
		# return the list containing the intermediate outputs
		return blockOutputs
	
class Decoder(nn.Module):
	def __init__(self, channels=(64, 32, 16)):
		super(Decoder, self).__init__()
		# initialize the number of channels, upsampler blocks, and
		# decoder blocks
		self.channels = channels
		
		self.upconvs = nn.ModuleList([nn.ConvTranspose2d(channels[i], channels[i + 1], 2, 2) for i in range(len(channels) - 1)])
		# kernal size = 2, stride = 2 means after every ConvTrans2d,
        # the image size becomes twice large in each dimension
		# e.g. 2 x 2 becomes 4 x 4
		
		self.dec_blocks = nn.ModuleList([Block(channels[i], channels[i + 1]) for i in range(len(channels) - 1)])
		
	def forward(self, x, encFeatures):
		# loop through the number of channels
		for i in range(len(self.channels) - 1):
			# pass the inputs through the upsampler blocks
			x = self.upconvs[i](x)
			# crop the current features from the encoder blocks,
			# concatenate them with the current upsampled features,
			# and pass the concatenated output through the current
			# decoder block
			encFeat = self.crop(encFeatures[i], x)
			x = torch.cat([x, encFeat], dim=1)
			# dim=1 means concatenate in the dimension of channel

			x = self.dec_blocks[i](x)
		# return the final decoder output
		return x
	
	def crop(self, encFeatures, x):
		# grab the dimensions of the inputs, and crop the encoder
		# features to match the dimensions
		(_, _, H, W) = x.shape
		# In CNN, this 4 dimensions are: batch size, in channel, height, width
		encFeatures = CenterCrop([H, W])(encFeatures)
		# return the cropped features
		return encFeatures
	
class UNet(nn.Module):
	def __init__(self, encChannels=(3, 16, 32, 64), decChannels=(64, 32, 16), nbClasses=1,
			retainDim=True, outSize=(config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH)):
		
		super(UNet, self).__init__()
		
		# initialize the encoder and decoder
		self.encoder = Encoder(encChannels)
		self.decoder = Decoder(decChannels)
		
		# initialize the regression head and store the class variables
		self.head = nn.Conv2d(decChannels[-1], nbClasses, 1)
		self.retainDim = retainDim
		self.outSize = outSize
		
	def forward(self, x):
        # grab the features from the encoder
		encFeatures = self.encoder(x)
		
        # pass the encoder features through decoder making sure that
        # their dimensions are suited for concatenation
		decFeatures = self.decoder(encFeatures[::-1][0], encFeatures[::-1][1:])
		# [::-1] means reverse the order of encFeatures
		
        # pass the decoder features through the regression head to
        # obtain the segmentation mask
		map = self.head(decFeatures)
		
        # check to see if we are retaining the original output
        # dimensions and if so, then resize the output to match them
		if self.retainDim:
			map = nn.functional.interpolate(map, self.outSize)
			
        # return the segmentation map
		return map