import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from modelf import ModelF
from modelg import ModelG
from modele import ModelE
from modeld import ModelD
from datasets import PointDataset

BATCH_SIZE = 5
POINTS_PER_SAMPLE = 100
LEARNING_RATE = 0.01
MOMENTUM = 0.9

f = ModelF()
g = ModelG()
e = ModelE()
d = ModelD()

sample_input = torch.rand(BATCH_SIZE*POINTS_PER_SAMPLE, 4).cuda()
w1 = f(sample_input)
fake_sdf = g(sample_input[:, :3], w1)
fake_psdf = torch.cat((sample_input[:,:3],fake_sdf), dim=1)
w2 = e(fake_psdf)
p_fake = d(w2)

# TODO: fine-tune the optimizers
foptim = optim.Adam(f.parameters(), lr=LEARNING_RATE)
goptim = optim.Adam(g.parameters(), lr=LEARNING_RATE)
eoptim = optim.Adam(e.parameters(), lr=LEARNING_RATE)
doptim = optim.Adam(d.parameters(), lr=LEARNING_RATE)

chair_ds = PointDataset.from_split('data/ShapeNet_SDF/chairs', 'train')
chair_loader = DataLoader(chair_ds, batch_size=BATCH_SIZE)

# Use explicit progress bar for displaying loss during training
pbar = tqdm(chair_loader, desc='Training model')
for chair_batch in pbar:
    uniform, surface = chair_batch
    flat_uniform = uniform.view(-1, 4).cuda()
    flat_uniform.requires_grad = True

    # Discriminator optimization step
    eoptim.zero_grad()
    doptim.zero_grad()
    w1 = f(flat_uniform)
    fake_sdf = g(flat_uniform[:, :3], w1)
    fake_psdf = torch.cat((flat_uniform[:, :3], fake_sdf), dim=1)
    w2 = e(fake_psdf)
    p_fake = d(w2)
    p_real = d(e(flat_uniform))
    discriminator_loss = torch.mean(-0.5*torch.log(p_real) - 0.5*torch.log(1-p_fake))
    discriminator_loss.backward()
    eoptim.step()
    doptim.step()
    pbar.set_postfix(dloss=discriminator_loss.item())

    # Generator optimization step
    foptim.zero_grad()
    goptim.zero_grad()
    w1 = f(flat_uniform)
    fake_sdf = g(flat_uniform[:, :3], w1)
    fake_psdf = torch.cat((flat_uniform[:, :3], fake_sdf), dim=1)
    w2 = e(fake_psdf)
    p_fake = d(w2)
    generator_loss = torch.mean(-0.5*torch.log(p_fake))
    generator_loss.backward()
    foptim.step()
    goptim.step()
    pbar.set_postfix(gloss=generator_loss.item())

    # Penalize the divergence between generator's and discriminator's latent
    # vectors, following Pidhorskyi et al. 2020
    # This ensures that generator and discriminator's intermediate
    # representations are in the same vector space
    foptim.zero_grad()
    eoptim.zero_grad()
    w1 = f(flat_uniform)
    fake_sdf = g(flat_uniform[:, :3], w1)
    fake_psdf = torch.cat((flat_uniform[:, :3], fake_sdf), dim=1)
    w2 = e(fake_psdf)
    latent_loss = torch.sum(0.5*(w1 - w2)**2)
    latent_loss.backward()
    foptim.step()
    eoptim.step()
    pbar.set_postfix(lloss=latent_loss.item())
    loss_status = f'discriminator loss: {discriminator_loss}, generator loss: {generator_loss}, latent loss: {latent_loss}'
    pbar.set_description(desc=loss_status)

torch.save(f, 'saved_models/f.torch')
torch.save(g, 'saved_models/g.torch')
torch.save(e, 'saved_models/e.torch')
torch.save(d, 'saved_models/d.torch')
