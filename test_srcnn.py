import torch
from network import SRCNN
from PIL import Image
from torchvision import datasets, transforms, utils as tvutils
from util import convert_rgb_to_y, convert_rgb_to_ycbcr, convert_ycbcr_to_rgb, calc_psnr
import numpy as np
import os


device = torch.device('mps')
model = SRCNN()
model.load_state_dict(torch.load("./srcnn_x4.pth", map_location = device))
model = model.to(device)

# model = SRCNN()
# model.load_state_dict(torch.load("./SRCNN_perception_loss.pth", map_location = device))
# model = model.to(device)

# model = torch.load("./SRCNN_perception_loss.pth", map_location = device)


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

total_psnr = 0

# for path in os.listdir("test"):
data = Image.open("./mai1.jpeg")
data = transform(data)
blurred_data = blur(data)
# print(test_image.shape)
blurred_data = np.array(blurred_data).astype(np.float32)
blurred_data = blurred_data.transpose(1, 2, 0)
blurred_data = (blurred_data*255.0).clip(0, 255).astype("uint8")
data = data.to(device)

# with torch.no_grad():
#     preds = model(blurred_data)

    # img = Image.fromarray(blurred_test_image)
    # img.save("./blurred_test/" + path)

    # img = np.array(original_image).astype(np.float32)
    # img = img.transpose(1, 2, 0)
    # img = ((img-16)*255.0).clip(0, 255).astype("uint8")
    # img = Image.fromarray(img[:, :, 0], 'L')
    # img.save("./test_1/" + path)

    # original_image = original_image.to(device)
    # blurred_test_image = blurred_test_image.to(device)

    # with torch.no_grad():
    #     preds = model(blurred_test_image).clamp(0.0, 1.0)
    # print(preds)

ycbcr = convert_rgb_to_ycbcr(blurred_data)
                # print(ycbcr.shape)
y = ycbcr[..., 0]
y /= 255.
y = y.astype(np.float32)
y = torch.from_numpy(y).to(device)
y = y.unsqueeze(0).unsqueeze(0)
# print(y.shape)

with torch.no_grad():
    preds = model(y).clamp(0.0, 1.0)

psnr = calc_psnr(data, preds)
print('PSNR: {:.2f}'.format(psnr))

preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
output = Image.fromarray(output)
output.save("pixel.jpg")

print(total_psnr / 50)
