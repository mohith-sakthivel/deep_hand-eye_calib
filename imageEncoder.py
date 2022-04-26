import torch
import torch.nn as nn
import os
from tqdm import tqdm
from PIL import Image
from glob import glob
from torchvision import transforms

# Transformation for inpnut into the encoder
transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), 
                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

# Import ResNet18 model with pretrained weights in evaluation mode
res18 = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
res18.eval()
for param in res18.parameters():
    param.requires_grad = False 

# Store all encoded features into features directory
cache_location = 'features/'

# Using rectified images with maximum brightness
image_dir = 'SampleSet/MVS Data/Rectified/{}/'
image_filename_pattern = 'rect_{:0>3}_max.png'
brightness = 'max'

# Obtain the list of scans in the rectified directory
dir_list = os.listdir('SampleSet/MVS Data/Rectified/')

# For each scan list
for dir in tqdm(dir_list):
    # Get all images that have maximum brightness
    img_list = glob(f'{image_dir.format(dir)}*{brightness}*')
    
    # For each image, open the image, transform, encode, and save as a torch pt file
    for image_id in range(1,len(img_list)+1):
        # the caching and loading logic here
        feat_path = os.path.join(cache_location, f'feat_{dir}_{image_id}.pt')
        image_path = os.path.join(image_dir.format(dir), image_filename_pattern.format(image_id))
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0)
        image = res18(image)[0]
        torch.save(image, feat_path)