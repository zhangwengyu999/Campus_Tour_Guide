#############################################################################
# Campus-tour Guide: On-campus Landmarks Recognition System (OLRS)          #
#                                                                           #
#                                                                           #
# This is the Algorithm program for the OLRS                                #
#                                                                           #
#                                                                           #
#############################################################################

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import transforms
from PIL import Image

IMG_SIZE = 128

class Sample:
    def __init__(self, idx=0, fname='', img=None, colorImg=None, feat=None, VGGfeat=None, label=None):
        self.idx = idx
        self.fname = fname
        self.img = img
        self.colorImg = colorImg
        self.feat = feat
        self.VGGfeat = VGGfeat
        self.label = label
        self.pred = None

def squareImg(image):
    # Get image semiaxes
    img_h_saxis = image.shape[0]//2
    img_w_saxis = image.shape[1]//2
    crop_saxis = min((img_h_saxis, img_w_saxis))
    center = (img_h_saxis, img_w_saxis)
    cropped_img = image[(center[0]-crop_saxis): (center[0]+ crop_saxis),
                        (center[1]-crop_saxis): (center[1]+ crop_saxis)]
    return cropped_img

def img2Sample(inImg):
    img = inImg
    colorImg = img[..., ::-1] #BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # grey scale

    colorImg = squareImg(colorImg)
    colorImg = cv2.resize(colorImg, (IMG_SIZE,IMG_SIZE))

    img = squareImg(img) # crop to square
    img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
    return Sample(0, None, img, colorImg, None, None, None)

def getCNNImgData(inImg):
    normalize = transforms.Normalize(
        mean=[0.456],
        std=[0.224]
    )
    preprocess = transforms.Compose([
        transforms.Resize([IMG_SIZE,IMG_SIZE]),
        transforms.ToTensor(),
        normalize
    ])
    img_tensor = preprocess(Image.fromarray(inImg))
    return img_tensor

# define CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # 128 * 128 * 1 => 128 * 128 * 32, 3x3 conv filter, padding 2
        self.conv1 = nn.Sequential(             # input shape (1, 128, 128)
            nn.Conv2d(
                in_channels=1,              
                out_channels=32,           
                kernel_size=5,              
                stride=1,                   
                padding=2,                      # same width and length after con2d, padding=(kernel_size-1)/2
            ),                                  # output shape (32, 128, 128)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),        # choose max value in 2x2 area, output shape (32, 64, 64)
        )
        
        self.conv2 = nn.Sequential(             # input shape (32, 64, 64)
            nn.Conv2d(32, 64, 5, 1, 2),         # output shape (64, 64, 64)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),        # output shape (64, 32, 32)
        )
        
        self.d1 = nn.Linear(64 * 32 * 32, 256)  # fully connected layer
        self.d2 = nn.Linear(256, 128)           # fully connected layer
        self.d3 = nn.Linear(128, 15)            # fully connected layer, output 15 classes
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1) # flatten the output of conv2 to (batch_size, 64 * 32 * 32)
        x = self.d1(x)
        x = F.relu(x)
        x = self.d2(x)
        x = F.relu(x)
        x = self.d3(x)
        output = self.sigmoid(x)

        return output

def predictCNN(inImg, inModel):
    imgSample = img2Sample(inImg)
    img = imgSample.img
    samData = getCNNImgData(img)
    samData = samData.view(1, 1, IMG_SIZE, IMG_SIZE)
    inModel.eval()
    pred = inModel(samData)
    prediction = int(torch.max(pred.data, 1)[1].numpy())
    return prediction
           
if __name__ == '__main__':
    model = torch.load('modelCNN.pth')
    print(predictCNN("112.jpeg",model))
