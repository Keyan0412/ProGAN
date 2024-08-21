import torch
from torchvision import transforms
import numpy as np
import os
import argparse

from network import Discriminator, Generator
from train import generate_16, load_checkpoint
from dataset import Mydata
from setting import IN_CHANNELS, IMAGE_SIZE, CHANNELS_IMG


# get argument
parser = argparse.ArgumentParser(
    description='Test your model and dataset', 
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
parser.add_argument('-s', '--step', type=int, required=True, help='step of training')
args = parser.parse_args()
step = args.step

# get parameters
device = "cuda" if torch.cuda.is_available() else "cpu"
netD = Discriminator(in_channels=IN_CHANNELS, img_size=IMAGE_SIZE, img_channels=CHANNELS_IMG).to(device)
netG = Generator(in_channels=IN_CHANNELS, img_size=IMAGE_SIZE, img_channels=CHANNELS_IMG).to(device)
if os.path.exists('workdir/model.pth.tar'):
    checkpoint = torch.load('workdir/model.pth.tar', weights_only=False)
    load_checkpoint(checkpoint=checkpoint, gen=netG, disc=netD)
else:
    print("you don't have saved model in your workplace. Ensure the name is 'model.pth.tar'.")

img_size = 4 * 2**step

'''Generator test'''
try:
    generate_16(netG, step=step, filename='images/test_output.png')
except torch.OutOfMemoryError:
    print('Warning: CUDA out of memory. Trying to use small batch to generate.')
    outputs = torch.randn((16, 3, img_size, img_size))
    with torch.no_grad():
        for i in range(16):
            z = random_tensor = torch.rand((1, 512, 1, 1)).to(device)
            try:
                output = netG(z, alpha=1, steps=step)
            except torch.OutOfMemoryError:
                print('Your device is not enough to generate such big image.')
                exit(-1)
            outputs[i] = output

    grid_img = torch.zeros((3, 4 * img_size, 4 * img_size))  
    for i, img in enumerate(outputs):  
        row = i // 4
        col = i % 4
        grid_img[:, row * img_size:(row + 1) * img_size, col * img_size:(col + 1) * img_size] = img 
    transforms.ToPILImage()(grid_img*0.5 + 0.5).save('images/test_output.png')

print('complete generation.')


'''Sampling test'''
transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5 for _ in range(CHANNELS_IMG)],
                [0.5 for _ in range(CHANNELS_IMG)],
            ),
        ]
    )
dataset = Mydata(dir='./data', transform=transform)

# generate random indices
indices = torch.randint(high=len(dataset), size=(16, ), requires_grad=False).tolist()
output = [dataset[i][0] for i in indices]
output = torch.stack(output)

# create grid
grid_size = int(np.sqrt(output.size(0)))   
grid_img = torch.zeros((3, grid_size * img_size, grid_size * img_size))  
  
# place images in the correct position
for i, img in enumerate(output):  
    row = i // grid_size  
    col = i % grid_size  
    grid_img[ : , row * img_size : (row + 1) * img_size, col * img_size : (col + 1) * img_size] = img

transforms.ToPILImage()(grid_img*0.5+0.5).save('images/test_sample.png')

print('The test images has been generated. Please see your /images folder.')
