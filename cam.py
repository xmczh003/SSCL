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
    img = img[:, :, ::-1]   				# 1
    img = np.ascontiguousarray(img)			# 2
    #img = Image.fromarray(img)
    transform = transforms.Compose([
        #transforms.Resize([248,248]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # transform = transforms.Compose([
    #     #transforms.Resize(248),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])
    img = transform(img)
    img = img.unsqueeze(0)					# 3
    return img

def apply_colormap_on_image(org_im, activation, colormap_name):
    """
        Apply heatmap on image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        colormap_name (str): Name of the colormap
    """
    # Get colormap
    #activation = activation.T
    color_map = mpl_color_map.get_cmap(colormap_name)
    no_trans_heatmap = color_map(activation)
    # Change alpha channel in colormap to make sure original image is displayed
    heatmap = copy.copy(no_trans_heatmap)
    heatmap[:, :, 3] = 0.4
    heatmap = Image.fromarray((heatmap*255).astype(np.uint8))
    no_trans_heatmap = Image.fromarray((no_trans_heatmap*255).astype(np.uint8))

    # Apply heatmap on iamge
    org_im = Image.fromarray(org_im.astype(np.uint8))
    #org_im = Image.fromarray(org_im.squeeze(0).reshape(248,248,3).numpy().astype(np.uint8))
    heatmap_on_image = Image.new("RGBA", org_im.size)
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, org_im.convert('RGBA'))
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
    return no_trans_heatmap, heatmap_on_image


def format_np_output(np_arr):
    """
        This is a (kind of) bandaid fix to streamline saving procedure.
        It converts all the outputs to the same format which is 3xWxH
        with using sucecssive if clauses.
    Args:
        im_as_arr (Numpy array): Matrix of shape 1xWxH or WxH or 3xWxH
    """
    # Phase/Case 1: The np arr only has 2 dimensions
    # Result: Add a dimension at the beginning
    if len(np_arr.shape) == 2:
        np_arr = np.expand_dims(np_arr, axis=0)
    # Phase/Case 2: Np arr has only 1 channel (assuming first dim is channel)
    # Result: Repeat first channel and convert 1xWxH to 3xWxH
    if np_arr.shape[0] == 1:
        np_arr = np.repeat(np_arr, 3, axis=0)
    # Phase/Case 3: Np arr is of shape 3xWxH
    # Result: Convert it to WxHx3 in order to make it saveable by PIL
    if np_arr.shape[0] == 3:
        np_arr = np_arr.transpose(1, 2, 0)
    # Phase/Case 4: NP arr is normalized between 0-1
    # Result: Multiply with 255 and change type to make it saveable by PIL
    if np.max(np_arr) <= 1:
        np_arr = (np_arr*255).astype(np.uint8)
    return np_arr


def save_image(im, path):
    """
        Saves a numpy matrix or PIL image as an image
    Args:
        im_as_arr (Numpy array): Matrix of shape DxWxH
        path (str): Path to the image
    """
    if isinstance(im, (np.ndarray, np.generic)):
        im = format_np_output(im)
        im = Image.fromarray(im)
    im.save(path)


def save_class_activation_images(org_img, activation_map, dir_name, file_name):
    """
        Saves cam activation map and activation map on the original image

    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        file_name (str): File name of the exported image
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    # Grayscale activation map
    heatmap, heatmap_on_image = apply_colormap_on_image(org_img, activation_map, 'rainbow')
    # Save colored heatmap
    # path_to_file = os.path.join(dir_name, file_name+'_Cam_Heatmap.png')
    # save_image(heatmap, path_to_file)
    # Save heatmap on iamge
    path_to_file = os.path.join(dir_name, file_name+'_Cam_On_Image.png')
    save_image(heatmap_on_image, path_to_file)
    # SAve grayscale heatmap
    # path_to_file = os.path.join(dir_name, file_name+'_Cam_Grayscale.png')
    # save_image(activation_map.T, path_to_file)


class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None

    def forward_pass(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)

        conv_output = x  # Save the convolution output on that layer

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.fc(x)

        return conv_output, x

    def get_weights(self, index):
        return self.model.fc.weight.data[index, :]


class Cam():
    """
        Produces class activation map
    """
    def __init__(self, model):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model)

    def generate_cam(self, input_image, target_class=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.extractor.forward_pass(input_image)
        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())
        weight_softmax = self.extractor.get_weights(target_class)
        weight_softmax = weight_softmax.data.numpy() # C
        conv_output = conv_output.data.squeeze(0).numpy() # C*H*W
        c, h, w = conv_output.shape
        cam = weight_softmax.dot(conv_output.reshape(c, h*w)) # (C)x(C*HW) -> HW
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        cam = np.uint8(255 * cam)
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                       input_image.shape[3]), Image.ANTIALIAS))#/255
        return cam


class GradCamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.avg = nn.AdaptiveAvgPool2d(1)

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass(self, x):
        # x = self.model.conv1(x)
        # x = self.model.bn1(x)
        # x = self.model.relu(x)
        # x = self.model.maxpool(x)
        x_raw = self.model.module.features_raw(x)
        x_coarse = self.model.module.features_coarse(x_raw)
        x = self.model.module.features(x_coarse)

        # x = self.model.layer1(x)
        # x = self.model.layer2(x)
        # x = self.model.layer3(x)

        x.register_hook(self.save_gradient)
        conv_output = x  # Save the convolution output on that layer
        x.register_hook(self.save_gradient)
        conv_output = x

        # x = self.model.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.model.fc(x)
        x = self.model.module.raw_classifier(self.avg(x).view(-1, 2048))
        return conv_output, x


class GradCam():
    """
        Produces class activation map
    """
    def __init__(self, model):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = GradCamExtractor(self.model)

    def generate_cam(self, input_image, target_class=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)

        # output = model(input_image, p=1)
        # conv_output = output[5]
        # model_output = output[0]
        conv_output, model_output = self.extractor.forward_pass(input_image)
        if target_class is None:
            target_class = np.argmax(model_output.cpu().data.numpy())
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Zero grads
        # self.model.features.zero_grad()
        # self.model.classifier.zero_grad()
        for m in self.model.modules():
            m.zero_grad()
        # Backward pass with specified target
        model_output.backward(gradient=one_hot_output.cuda(), retain_graph=True)
        # Get hooked gradients
        guided_gradients = self.extractor.gradients.data.cpu().numpy()[0]
        # Get convolution outputs
        target = conv_output.data.cpu().numpy()[0]
        # Get weights from gradients
        weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient
        # Create empty numpy array for cam
        # cam = np.ones(target.shape[1:], dtype=np.float32)
        cam = np.zeros(target.shape[1:], dtype=np.float32)
        # Multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                       input_image.shape[3]), Image.ANTIALIAS))#/255
        return cam

if __name__ == '__main__':
    model = s3n()
    model = nn.DataParallel(model)
    device = torch.device('cuda')
    params = model.parameters()
    resume = './model_10.pt'
    checkpoint = torch.load(resume)
    model.load_state_dict(checkpoint['model'])
    gradCAM = GradCam(model=model)
    f = open('./datasets/Flavia/images.txt', 'r')
    lines = f.readlines()
    for line in lines:
        root = './datasets/Flavia/images/'
        line = line.split()[0]
        name = line.split('/')[-1]
        images_path = os.path.join(root, line)
        #images_path = '1259110850_0053.jpg'
        #print(line, name, images_path)
        images = cv2.imread(images_path)
        images = cv2.resize(images,(248,248))
        img = img_preprocess(images)
        cam_output = gradCAM.generate_cam(img.cuda())
        #cam_output = Cam.generate_cam(images.cuda())
        save_class_activation_images(images, cam_output, './CAM/3', name)
        print(line)

