import torch
import torchvision.utils as vutils


def write_images(image_list, nrow, file_name):
    '''
    image_list: a list of tensor (image), they are assumed to have the same width and height
    nrow: the number of images to be shown for each row, i.e. the number of column

    '''
    image_list = [x.expand(1, 3, -1, -1) for x in image_list] # expand gray-scale images to 3 channels
    image_list = torch.cat(image_list) # Nx3xHxW
    image_grid = vutils.make_grid(image_list, nrow=nrow, padding=0, normalize=True)
    vutils.save_image(image_grid, file_name, nrow=1)

def write_images2(image_list, file_name):
    image_grid = image_list[0]/4+image_list[1]/4+image_list[2]/2
    vutils.save_image(image_grid, file_name, nrow=1)

if __name__ == '__main__':
    import os
    from PIL import Image
    import torchvision.transforms as T

    root = './CAM'
    original_path = os.path.join(root,'1')
    sampling_D_path = os.path.join(root, '2')
    sampling_C_path = os.path.join(root, '3')
    img_list = os.listdir(original_path)
    img_list = [x for x in img_list if x.endswith('png')]
    i = 0
    path = './grid'
    for i in range(int(len(img_list)/1)):
        tem_list = img_list[1*i : 1*(i+1)]
        image_list = []
        for x in tem_list:
            image_list.append(T.ToTensor()(Image.open(original_path + '/' + x)))
            image_list.append(T.ToTensor()(Image.open(sampling_D_path + '/' + x)))
            image_list.append(T.ToTensor()(Image.open(sampling_C_path + '/' + x)))
        grid_path = os.path.join(path, x + '.png')
        write_images2(image_list, grid_path)
        i += 1

