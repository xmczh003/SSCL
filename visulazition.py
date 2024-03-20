import cv2
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import models
import json
from sss_net import s3n


def img_preprocess(img_in):
    img = img_in.copy()
    img = img[:, :, ::-1]   				# 1
    img = np.ascontiguousarray(img)			# 2
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4948052, 0.48568845, 0.44682974], [0.24580306, 0.24236229, 0.2603115])
    ])
    img = transform(img)
    img = img.unsqueeze(0)					# 3
    return img

def backward_hook(module, grad_in, grad_out):
    grad_block.append(grad_out[0].detach())

def farward_hook(module, input, output):
    fmap_block.append(output)

def cam_show_img(img, feature_map, grads, out_dir):
    H, W, _ = img.shape
    cam = np.zeros(feature_map.shape[1:], dtype=np.float32)		# 4
    grads = grads.reshape([grads.shape[0],-1])					# 5
    weights = np.mean(grads, axis=1)							# 6
    for i, w in enumerate(weights):
        cam += w * feature_map[i, :, :]							# 7
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    cam = cv2.resize(cam, (W, H))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    cam_img = 0.3 * heatmap + 0.7 * img

    path_cam_img = os.path.join(out_dir, "cam.jpg")
    cv2.imwrite(path_cam_img, cam_img)


if __name__ == '__main__':
    path_img = '001.Black_footed_Albatross/Black_Footed_Albatross_0002_55.jpg'
    json_path = './cam/labels.json'
    output_dir = './cam'

    # with open(json_path, 'r') as load_f:
    #     load_json = json.load(load_f)
    # classes = {int(key): value for (key, value)
    #            in load_json.items()}

    #classes = list(classes.get(key) for key in range(1000))

    fmap_block = list()
    grad_block = list()

    img = cv2.imread(path_img, 1)
    img_input = img_preprocess(img)
    net = s3n()
    net = torch.nn.DataParallel(net)
    device = torch.device('cuda')
    params = net.parameters()
    resume = './generate/s3n/model_40.pt'
    checkpoint = torch.load(resume)
    net.load_state_dict(checkpoint['model'])
    net.eval()

    # forward
    output = net(img_input)
    idx = np.argmax(output.cpu().data.numpy())

    # backward
    net.zero_grad()
    class_loss = output[0, idx]
    class_loss.backward()

    grads_val = grad_block[0].cpu().data.numpy().squeeze()
    fmap = fmap_block[0].cpu().data.numpy().squeeze()

    cam_show_img(img, fmap, grads_val, output_dir)