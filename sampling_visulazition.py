import os, copy
import numpy as np
from PIL import Image
import matplotlib.cm as mpl_color_map
from torch import nn
import torch
from sss_net import s3n
import torchvision.transforms as transforms
import cv2

def img_preprocess(img_in):
    img = img_in.copy()
    img = img[:, :, ::-1]
    img = np.ascontiguousarray(img)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    img = transform(img)
    img = img.unsqueeze(0)
    return img


def save_image_tensor2pillow(input_tensor: torch.Tensor, filename):

    #assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)

    input_tensor = input_tensor.clone().detach()

    input_tensor = input_tensor.to(torch.device('cpu'))

    input_tensor = input_tensor.squeeze()
    input_tensor = input_tensor.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
    im = Image.fromarray(input_tensor)
    im.save(filename)

def save_tensor_images(tensor_images, dir_name, file_name):
    """
        Saves cam activation map and activation map on the original image

    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        file_name (str): File name of the exported image
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    show = transforms.ToPILImage()
    path_to_file = os.path.join(dir_name, file_name)
    show(tensor_images.squeeze(0)).save(path_to_file)

if __name__ == '__main__':
    model = s3n()
    model = nn.DataParallel(model)
    device = torch.device('cuda')
    params = model.parameters()
    resume = './model_10.pt'
    checkpoint = torch.load(resume)
    model.load_state_dict(checkpoint['model'])
    f = open('./datasets/Flavia/images.txt', 'r')
    lines = f.readlines()
    for line in lines:
        root = './datasets/Flavia/images/'
        line = line.split()[0]
        name = line.split('/')[-1]
        images_path = os.path.join(root, line)
        #images_path = '1259110850_0053.jpg'
        images = cv2.imread(images_path)
        images = cv2.resize(images, (248, 248))
        img = img_preprocess(images)
        sampling_D, sampling_C = model(img.cuda(), p=2)[-2:]
        save_tensor_images(sampling_D, 'sampling/sampling_D', name)
        save_tensor_images(sampling_C, 'sampling/sampling_C', name)
        img = transforms.Resize([448,448])(img)
        save_tensor_images(img, 'sampling/original', name)
        print(line)
    f.close()

