from network import SRCNN
import torch

import torch
import torch.nn as nn
from vgg import vgg16
from network import SRCNN
from loss import get_loss, per_pixel_loss
import os

# from torchmetrics.image import StructuralSimilarityIndexMeasure

import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, utils as tvutils
from PIL import Image
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import logging
import time
from io import BytesIO
from tqdm import tqdm
from util import convert_rgb_to_y, convert_rgb_to_ycbcr, convert_ycbcr_to_rgb, calc_psnr


BATCH_SIZE = 4
NUM_ITERATION = 40000
LEARNING_RATE = 1e-4
NUM_EPOCHES = 1
LOG_EVERY = 200
SAMPLES_EVERY = 1000
STEP_LR = 2e-3

def setup_logging():
    log_formatter = logging.Formatter(
        '%(asctime)s: %(levelname)s %(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logging.getLogger().handlers = []
    if not len(logging.getLogger().handlers): 
        logging.getLogger().addHandler(console_handler)
    logging.getLogger().setLevel(logging.INFO)

def logger(tag, value, global_step):
    if tag == '':
       logging.info('')
    else:
       logging.info(f'  {tag:>8s} [{global_step:07d}]: {value:5f}')

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def save_image(data, save_dir):
    std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    img = data.detach().clone().cpu().numpy()[0]
    img = ((img * std + mean).transpose(1, 2, 0)*255.0).clip(0, 255).astype("uint8")
    img = Image.fromarray(img)
    img.save(save_dir)

def train_model():
    device = torch.device('mps')
    # setup_logging()
    torch.set_num_threads(4)
    writer = SummaryWriter("./log.txt", max_queue=1000, flush_secs=120)

    model = SRCNN()
    model.load_state_dict(torch.load("./srcnn_x4.pth", map_location = device))
    model = model.to(device)

    transform = transforms.Compose([
        transforms.Resize(288),           
        transforms.CenterCrop(288),      
        transforms.ToTensor()
    ])

    blur = transforms.Compose([    
        transforms.Resize(72),       
        transforms.GaussianBlur(1),
        transforms.Resize(288)
    ])
    
    dataset = datasets.ImageFolder("/Users/weilai/Desktop/UIUC/FA24/CS444/project/coco", transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


    test_image = Image.open("./mai1.jpeg")
    test_image = transform(test_image)

    blurred_test_image = blur(test_image)
    # print(test_image.shape)
    blurred_test_image = np.array(blurred_test_image).astype(np.float32)
    blurred_test_image = blurred_test_image.transpose(1, 2, 0)
    blurred_test_image = (blurred_test_image*255.0).clip(0, 255).astype("uint8")
    test_image = test_image.to(device)

    img = Image.fromarray(blurred_test_image)
    img.save("blurred_test_image.jpg")

    vgg = vgg16()
    vgg = vgg.to(device)

    iterations = 0
    train_loss = []
    total_psnr = []
    total_percept_loss = 0
    total_pixel_loss = 0
    for epoch in range(1, NUM_EPOCHES + 1):
        batch_idx = 0
        for data, _ in tqdm(dataloader, desc=f"Epoch {epoch}", unit="batch"):
            data = convert_rgb_to_y(data).unsqueeze(dim = 1)
            # print(data.shape)
            blurred_data = blur(data)
            blurred_data = blurred_data.to(device)
            data = data.to(device)

            batch_idx = batch_idx + 1
            optimizer.zero_grad()

            y_hat = model(blurred_data)
            loss = per_pixel_loss(y_hat, data)
            total_pixel_loss += loss
            
            # y_hat = y_hat.repeat((1, 3, 1, 1)).to(device)
            # y_hat_features = vgg(y_hat)
            # data_1 = data.repeat((1, 3, 1, 1)).to(device)
            # x_features = vgg(data_1)

            # loss = get_loss(y_hat_features, x_features)
            # loss += percept_loss
            # total_percept_loss += percept_loss

            loss.backward()
            optimizer.step()

            train_loss += [loss.item()]
            iterations += 1
            current_psnr = calc_psnr(data, y_hat)
            total_psnr += [current_psnr.item()]
            # print(iterations)

            if iterations % LOG_EVERY == 0:
                writer.add_scalar('loss', np.mean(train_loss), iterations)
                print('loss', np.mean(train_loss), np.mean(total_psnr), iterations, loss.item(), total_pixel_loss.item(), current_psnr.item())
                train_loss = []
                total_psnr = []
                total_pixel_loss = 0
                total_percept_loss = 0

            if iterations % SAMPLES_EVERY == 0:
                model.eval()
                if not os.path.exists("visualization"):
                    os.makedirs("visualization")

                ycbcr = convert_rgb_to_ycbcr(blurred_test_image)
                # print(ycbcr.shape)
                y = ycbcr[..., 0]
                y /= 255.
                y = y.astype(np.float32)
                y = torch.from_numpy(y).to(device)
                y = y.unsqueeze(0).unsqueeze(0)
                # print(y.shape)

                with torch.no_grad():
                    preds = model(y).clamp(0.0, 1.0)

                psnr = calc_psnr(test_image, preds)
                print('PSNR: {:.2f}'.format(psnr))

                preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

                output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
                output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
                output = Image.fromarray(output)
                out_path = "visualization/mai_srcnn_x{}_{}.jpg".format(4, batch_idx)
                output.save(out_path)
                        
            if iterations >= 200000:
                break
    
    torch.save(model, 'SR_pixel_loss.pth')
                


