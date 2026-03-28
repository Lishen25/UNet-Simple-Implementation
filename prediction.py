# %%
# USAGE
# python predict.py
# import the necessary packages
from pyimagesearch import config
from pyimagesearch import model
from pyimagesearch import dataset
from imutils import paths # --> THIS PACKAGE IS TO MAKE OPENCV OPERATION MORE CONVENIENT
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm
import numpy as np
import torch
import os

# %%
# THIS FUNCTION IS FOR THE IMPLEMENTATION OF RLE ENCODING
def rle_encode(mask):
    pixels = mask.flatten(order='F')  # 等价于 mask.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# %%
# load the test data
print("[INFO] loading up test image paths...")
imagePaths = sorted(list(paths.list_images(config.TEST_IMAGE_DATASET_PATH)))
print(f"[INFO] found {len(imagePaths)} examples in the test set...")

# get a list for image ids
imageIDs = [os.path.splitext(os.path.basename(image))[0] for image in imagePaths]
# .basename() gets the whole file name, .splitext() splits into file name and extension

# %%
# load our model from disk and flash it to the current device
print("[INFO] load up model...")
unet = model.UNet(outSize=(config.OUT_IMAGE_HEIGHT, config.OUT_IMAGE_WIDTH)).to(config.DEVICE)
unet.load_state_dict(torch.load(config.KAGGLE_MODEL_PATH, weights_only=True))
unet.eval()

# %% 
# Test DataLoader
TestTransforms = transforms.Compose([transforms.ToPILImage(), 
                                 transforms.Resize((config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH)), 
                                 transforms.ToTensor()])

TestDS = dataset.TestDataset(imagePaths, transforms=TestTransforms)
TestLoader = DataLoader(TestDS, batch_size=config.BATCH_SIZE, shuffle=False, 
                        pin_memory=config.PIN_MEMORY, num_workers=os.cpu_count())

# %%
# Initialize a list to store the predictions.
predictions = []

# Iterate the testing set by batches.
for image in tqdm(TestLoader):
    imgs = image.to(config.DEVICE)

    # We don't need gradient in testing, and we don't even have labels to compute loss.
    # Using torch.no_grad() accelerates the forward process.
    with torch.no_grad():
        logits = unet(imgs)
        predMask = torch.sigmoid(logits)

        predMask = predMask.cpu().numpy()
        predMask = (predMask > config.THRESHOLD) * 255
        predMask = predMask.astype(np.uint8)
        
        for i in range(len(imgs)):
            rle_pred = rle_encode(predMask[i].squeeze(0))
            predictions.append(rle_pred)

# %%
# Save predictions into the file.
with open(config.PRED_PATHS, "w") as f:

    # The first row must be "od, rle_mask"
    f.write("id,rle_mask\n")

    # For the rest of the rows, each image id corresponds to a predicted class.
    for id, pred in zip(imageIDs, predictions):
         f.write(f"{id},{pred}\n")
# %%
