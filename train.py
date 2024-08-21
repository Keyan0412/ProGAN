import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from math import log2
import time
from tqdm import tqdm

from network import Generator, Discriminator
from dataset import Mydata
from setting import LEARNING_RATE_D, LAMBDA_GP, LEARNING_RATE_G, IMAGE_SIZE, \
                    CHANNELS_IMG, Z_DIM, IN_CHANNELS, BATCH_SIZES, \
                    NUM_WORKERS


torch.backends.cudnn.benchmarks = True
device = "cuda" if torch.cuda.is_available() else "cpu"

def gradient_penalty(critic, real, fake, alpha, train_step, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    beta = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * beta + fake * (1 - beta)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images, alpha, train_step)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


def save_checkpoint(state, filename="model.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, gen, disc, opt_gen=None, opt_disc=None):
    print("=> Loading checkpoint")
    gen.load_state_dict(checkpoint['gen'])
    disc.load_state_dict(checkpoint['critic'])

    if opt_gen != None and opt_disc != None:
        opt_gen.load_state_dict(checkpoint['opt_gen'])
        opt_disc.load_state_dict(checkpoint['opt_critic'])


def generate_16(netG, step, filename):
    z = random_tensor = torch.rand((16, 512, 1, 1)).to(device)
    with torch.no_grad():
        output = netG(z, alpha=1, steps=step)
    
    # calculate the size of grid 
    grid_size = int(np.sqrt(output.size(0)))  
    assert grid_size * grid_size == output.size(0), "批次大小应该是完全平方数"
    
    # create a big grid to place images
    grid_img = torch.zeros((3, grid_size * 4 * 2 ** step, grid_size * 4 * 2 ** step))  
    for i, img in enumerate(output):  
        row = i // grid_size  
        col = i % grid_size  
        grid_img[:, row * 4 * 2**step:(row + 1) * 4 * 2**step, col * 4 * 2**step:(col + 1) * 4 * 2**step] = img 
    transforms.ToPILImage()(grid_img*0.5 + 0.5).save(filename)

    return None


def train_fn(
    critic, gen,
    loader, dataset,
    step, alpha,
    opt_critic, opt_gen,
    num_epochs, batch_size
):
    start = time.time()
    total_time = 0
    num_batches = len(dataset) // batch_size + 1
    true_losses = np.zeros(num_batches, dtype=float)
    fake_losses = np.zeros(num_batches, dtype=float)
    losses_gen = np.zeros(num_batches, dtype=float)
    for batch_idx, (real, _) in enumerate(tqdm(loader, leave=True)):
        real = real.to(device)
        cur_batch_size = real.shape[0]
        model_start = time.time()

        # Train Critic: max E[critic(real)] - E[critic(fake)]
        # which is equivalent to minimizing the negative of the expression
        critic.zero_grad()
        noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(device)
        fake = gen(noise, alpha, step)

        # record
        critic_real = critic(real, alpha, step).reshape(-1)
        critic_fake = critic(fake, alpha, step).reshape(-1)
        true_loss = -torch.mean(critic_real)
        true_losses[batch_idx] = true_loss
        fake_loss = torch.mean(critic_fake)
        fake_losses[batch_idx] = fake_loss

        # step
        gp = gradient_penalty(critic, real, fake, alpha, step, device=device)
        pure_loss = true_loss + fake_loss
        loss_critic = pure_loss + LAMBDA_GP * gp
        loss_critic.backward(retain_graph=True)
        opt_critic.step()

        # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
        gen.zero_grad()
        fake = gen(noise, alpha, step)
        gen_fake = critic(fake, alpha, step).reshape(-1)
        loss_gen = -torch.mean(gen_fake)
        losses_gen[batch_idx] = loss_gen
        loss_gen.backward()
        opt_gen.step()

        # Update alpha and ensure less than 1
        alpha += cur_batch_size / (num_epochs * 0.5 * len(dataset))
        alpha = min(alpha, 1)

        total_time += time.time()-model_start

    print(f'Fraction spent on model training: {total_time/(time.time()-start)}')
    print('Loss of Discriminator:')
    print(f'-E[critic(real)]: {true_losses.mean()}')
    print(f' E[critic(fake)]: {fake_losses.mean()}')
    print('Loss of Generator: ')
    print(f'-E[critic(fake)]: {losses_gen.mean()}')
    print(f'current alpha: {alpha}')

    return alpha


def main():
    print(f'current device: {device}')
    gen = Generator(Z_DIM, IN_CHANNELS, img_size=IMAGE_SIZE, img_channels=CHANNELS_IMG).to(device)
    critic = Discriminator(IMAGE_SIZE, Z_DIM, IN_CHANNELS, img_channels=CHANNELS_IMG).to(device)

    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE_G, betas=(0.0, 0.99))
    opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE_D, betas=(0.0, 0.99))

    if not NEW_MODEL:
        checkpoint = torch.load('workdir/model.pth.tar', weights_only=True)
        load_checkpoint(checkpoint=checkpoint, gen=gen, disc=critic, opt_gen=opt_gen, opt_disc=opt_critic)

    gen.train()
    critic.train()

    step = STEP
    alpha = INITIAL_ALPHA

    num_epochs = NUM_EPOCH
    batch_size = BATCH_SIZES[step]
    TRANSFORM = transforms.Compose(
        [
            transforms.Resize((4 * 2**step, 4 * 2**step)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5 for _ in range(CHANNELS_IMG)],
                [0.5 for _ in range(CHANNELS_IMG)],
            ),
        ]
    )
    dataset = Mydata(dir='./data', transform=TRANSFORM)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

    for epoch in range(num_epochs):
        print(f'Eposh: {epoch+1}/{num_epochs}')
        alpha = train_fn(critic=critic, gen=gen, loader=loader, dataset=dataset, step=step, alpha=alpha, 
                 opt_critic=opt_critic, opt_gen=opt_gen, num_epochs=num_epochs, batch_size=BATCH_SIZES[step])

        # save records
        checkpoint = {
            'gen': gen.state_dict(),
            'critic': critic.state_dict(),
            'opt_gen': opt_gen.state_dict(),
            'opt_critic': opt_critic.state_dict(),
            }
        timestamp = time.time()  
        filename = 'save/' + time.strftime('%m%d|%H:%M', time.localtime(timestamp)) + '.pth.tar'
        save_checkpoint(checkpoint, filename=filename)
        
        # save in workplace
        destination_path = 'workdir/model.pth.tar'  
        with open(filename, 'rb') as source_file:  
            with open(destination_path, 'wb') as destination_file:  
                destination_file.write(source_file.read())

        # generate an example
        with torch.no_grad():
            gen.eval()
            timestamp = time.time() 
            filename = 'images/' + time.strftime('%m%d|%H:%M', time.localtime(timestamp)) + '.png' 
            generate_16(gen, step=step, filename=filename)
            gen.train()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='This is a program for learning parser', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument('-s', '--step', type=int, required=True, help='step of training')
    parser.add_argument('-e', '--epoch', type=int, help='number of epoch', default=20)
    parser.add_argument('-f', '--fade', action='store_true', help='using fade in')
    parser.add_argument('-n', '--new', action='store_true', help='train new model')

    args = parser.parse_args()

    NUM_EPOCH = args.epoch
    STEP = args.step
    INITIAL_ALPHA = 0.01 if args.fade else 1
    NEW_MODEL = args.new

    main()