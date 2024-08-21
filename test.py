import torch
from torchvision import transforms
import numpy as np

from network import Discriminator, Generator
from train import generate_16, get_loader, load_checkpoint


step = 6

device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint = torch.load('workdir/model.pth.tar', weights_only=False)
netD = Discriminator(in_channels=256).to(device)
netG = Generator(in_channels=256).to(device)
load_checkpoint(checkpoint=checkpoint, gen=netG, disc=netD)


'''Generator test'''
generate_16(netG, alpha=1, step=step, filename='images/test_output.png')


'''Sampling test'''
loader, dataset = get_loader(4 * 2**step)

# generate random indices
indices = torch.randint(high=19999, size=(16, ), requires_grad=False).tolist()
output = [dataset[i][0] for i in indices]
output = torch.stack(output)

# calculate size of grid
grid_size = int(np.sqrt(output.size(0)))  
assert grid_size * grid_size == output.size(0), "批次大小应该是完全平方数"  
  
# create grid to place images 
grid_img = torch.zeros((3, grid_size * 4 * 2 ** step, grid_size * 4 * 2 ** step))  
  
# place images in the correct position
for i, img in enumerate(output):  
    row = i // grid_size  
    col = i % grid_size  
    grid_img[:, row * 4 * 2**step:(row + 1) * 4 * 2**step, col * 4 * 2**step:(col + 1) * 4 * 2**step] = img

transforms.ToPILImage()(grid_img*0.5+0.5).save('images/test_sample.png')
