import os
import io
from torch.utils.data import Dataset
import h5py
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import grad as torch_grad

import torch.nn.utils.spectral_norm as spectral_norm
from utils import CondBatchNorm2d, init_weights, ResidualBlock, CriticFirstBlock, _gradient_penalty

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
num_train_D = 1

def hinge_loss_real(critic_real, _):
    loss_critic = (torch.nn.ReLU()(1.0 - critic_real)).mean()      # E[min(0, - Lip Const + D(x_real))]
    return loss_critic
    
def hinge_loss_fake(critic_fake, _):
    loss_critic = (torch.nn.ReLU()(1.0 + critic_fake)).mean()    # E[min(0, - Lip Const - D(x_fake))]
    return loss_critic
    
def G_loss(gen_fake, _):    
    return -torch.mean(gen_fake)

def L2_loss(Y, Y_):
    return torch.mean(((Y - Y_)**2).sum(axis=1)) #
	  

class Text2ImageDataset(Dataset):
    def __init__(self, dataset_dir, transform=None, split=0):
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.dataset = None
        self.dataset_keys = None
        self.split = 'train' if split == 0 else 'valid' if split == 1 else 'test'
        self.h5pyint = lambda x:int(np.array(x))
        
    def __len__(self):
        f = h5py.File(self.dataset_dir, 'r')
        self.dataset_keys = [str(k) for k in f[self.split].keys()]
        length = len(f[self.split])
        f.close()
        return length
    
    def __getitem__(self, idx):
        if self.dataset is None:
            self.dataset = h5py.File(self.dataset_dir, mode='r')
            self.dataset_keys = [str(k) for k in self.dataset[self.split].keys()]
        
        example_name = self.dataset_keys[idx]
        example = self.dataset[self.split][example_name]
        
        right_image = bytes(np.array(example['img']))
        right_embed = np.array(example['embeddings'], dtype=float)
        wrong_image = bytes(np.array(self.find_wrong_image(example['class'])))
        inter_embed = np.array(self.find_inter_embed())
        
        right_image = Image.open(io.BytesIO(right_image)).resize((64, 64))
        right_image = self.validate_image(right_image)
        wrong_image = Image.open(io.BytesIO(wrong_image)).resize((64, 64))
        wrong_image = self.validate_image(wrong_image)
        utf8_type = h5py.string_dtype('utf-8', 1000)
        txt = np.array(example['txt']).astype(utf8_type)
        
        sample = {
            'right_images' : torch.FloatTensor(right_image),
            'right_embed' : torch.FloatTensor(right_embed),
            'wrong_images' : torch.FloatTensor(wrong_image),
            'inter_embed' : torch.FloatTensor(inter_embed),
            'txt': str(txt)
            }
        sample['right_images'] = sample['right_images'].sub_(127.5).div_(127.5)
        sample['wrong_images'] = sample['wrong_images'].sub_(127.5).div_(127.5)

        return sample
    
    def validate_image(self, img):
        img = np.array(img, dtype=float)
        if len(img.shape) < 3:
            rgb = np.empty((64, 64, 3), dtype=np.float32)
            rgb[:, :, 0] = img
            rgb[:, :, 1] = img
            rgb[:, :, 2] = img
            img = rgb
        return img.transpose(2, 0, 1)

    def find_wrong_image(self, category):
        idx = np.random.randint(len(self.dataset_keys))
        example_name = self.dataset_keys[idx]
        example = self.dataset[self.split][example_name]
        _category = example['class']

        if _category != category:
            return example['img']

        return self.find_wrong_image(category)

    def find_inter_embed(self):
        idx = np.random.randint(len(self.dataset_keys))
        example_name = self.dataset_keys[idx]
        example = self.dataset[self.split][example_name]
        return example['embeddings']


class Trainer(object):
    def __init__(self, dataset_path, checkpoint_dir, output_dir, ngpu, num_workers, batch_size, image_size, 
                 nc, nz, nemb, ngf, ndf, lr, num_epochs, beta1, save_model_interval, test_interval, npics=4):

        dataset = Text2ImageDataset(dataset_path,
                                    transform=transforms.Compose([
                                        transforms.Resize(image_size),
                                        transforms.CenterCrop(image_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    ]))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        #Data_Augmentation
        dataset_c = Text2ImageDataset(dataset_path,
                                    transform=transforms.Compose([
                                        transforms.Resize(image_size),
                                        transforms.CenterCrop(image_size),
                                        transforms.RandomHorizontalFlip(0.5),
                                        transforms.RandomRotation((-20., 20.)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    ]))
        dataloader_c = torch.utils.data.DataLoader(dataset_c, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        dataloader_iterator = iter(dataloader_c)

        data_iterator = iter(dataloader)
        sample = next(data_iterator)

        fig = plt.figure()

        def multiple_imshow(fig, sample, idx):
            out = sample['right_images'][idx]
            out = out.data.mul_(127.5).add_(127.5).permute(1, 2, 0).byte().cpu().numpy()
            ax = fig.add_subplot(1, npics, idx+1)
            plt.imshow(out)

        final_title = ''
        for i in range(npics):
            multiple_imshow(fig, sample, i)
            final_title += str(sample['txt'][i]) + '\n'

        #plt.show()

        class Generator(nn.Module):
            def __init__(self, ngpu):
                super(Generator, self).__init__()
                self.ngpu = ngpu
                self.ngf = 64
                self.nc = 3
                self.nemb = 1024
                self.nproj = 25
                self.nz = 100
                
                self.num_residual = 4
                base_dim = 96

                self.projection = spectral_norm(nn.Linear(self.nemb, 128))
                self.projection2 = spectral_norm(nn.Linear(128, self.nproj, bias=False))
                torch.nn.init.orthogonal_(self.projection.weight)
                torch.nn.init.orthogonal_(self.projection2.weight)

                # First convolution layer for sampled latent vector z
                self.conv_first = spectral_norm(nn.Linear(in_features= int(self.nz/4), out_features= min(1024, base_dim*(2**(self.num_residual)))  * 4 * 4, bias=False)) #
                
                # 매개변수 초기화
                torch.nn.init.orthogonal_(self.conv_first.weight) 
                
                # Make residual blocks
                for i in range(self.num_residual):
                    self.add_module('residual_0' + str(i + 1), ResidualBlock(
                                                                in_channels= min(1024, base_dim*(2**(self.num_residual-i))),
                                                                out_channels= min(1024, base_dim*(2**(self.num_residual-i-1))),
                                                                kernel_size= 3,
                                                                stride= 1,
                                                                padding= 1,
                                                                resample= 'up'
                                                                ))

                # Make high resolution imgs
                # Upsampling
                # Batchnorm On/Off Switch will be needed


                self.output_hr = nn.Sequential(

                            nn.BatchNorm2d(base_dim, affine=False, momentum=None),
                            nn.ReLU(inplace= True),
                            spectral_norm(nn.Conv2d(    in_channels= base_dim, out_channels= 3,
                                      kernel_size= 3, stride= 1, padding= 1)),
                            nn.Tanh(),

                            )
                self.output_hr.apply(init_weights)


            def forward(self, z, emb):
                projected_emb = F.relu(self.projection(emb))
                projected_emb = torch.tanh(self.projection2(projected_emb))
                z_list = [z[:,0:int(z.size(1)/4)],z[:,int(z.size(1)/4):2*int(z.size(1)/4)],z[:,2*int(z.size(1)/4):3*int(z.size(1)/4)],z[:,3*int(z.size(1)/4):]] ## ??
                out_gen = self.conv_first(z_list[0]).reshape(z.size(0), -1, 4, 4)

                for i in range(self.num_residual):
                    layer_name = 'residual_0' + str(i + 1)

                    for name, module in self.named_children():
                        if layer_name == name:
                            if i != self.num_residual-1:
                                out_gen = module(out_gen, projected_emb * z_list[i+1])
                            else:
                                out_gen = module(out_gen, projected_emb)

                out_hr = self.output_hr(out_gen)
                return out_hr

        class Discriminator(nn.Module):
            def __init__(self, ngpu):
                super(Discriminator, self).__init__()
                self.ngpu = ngpu
                self.ndf = 64
                self.nc = 3
                self.nemb = 1024
                self.nproj = 32

                self.num_residual = 4
                base_dim = 96

                # First residual block

                self.conv_first = CriticFirstBlock(
                    in_channels=3,
                    out_channels=base_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    spec_norm=True,
                )

                # Make residual block
                for i in range(self.num_residual):

                    if i == 0 or i == 1 or i == 2:
                        resample = 'down'
                    else:
                        resample = None

                    self.add_module('residual_0' + str(i + 1),
                                    ResidualBlock(
                                        in_channels=min(1024, base_dim*(2**i)),
                                        out_channels=min(1024, base_dim*(2**(i+1))),
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        batch_norm=False,
                                        spec_norm=True,
                                        resample=resample,
                                    ))
                    ##why??
                    self.projection = spectral_norm(nn.Linear(self.nemb, 128))
                    self.projection2 = spectral_norm(nn.Linear(128, self.nproj, bias=False))

                    self.fc = spectral_norm(nn.Linear(min(1024, base_dim*2**self.num_residual), 1, bias=False))
                    torch.nn.init.orthogonal_(self.fc.weight)       #uniform 다른점?

            def forward(self, x, labels):
                out_critic = self.conv_first(x)

                emb = self.projection2(F.relu(self.projection(labels)))

                for i in range(self.num_residual):
                    layer_name = 'residual_0' + str(i + 1)
                    for name, module in self.named_children():
                        if layer_name == name:
                            out_critic = module(out_critic, emb)
                            # input [NCHW] = [N x BASE_DIM x 8 x 8],
                out_features = torch.sum(F.relu(out_critic, inplace=True), dim=(2, 3))  # output[NCHW] = [N x BASE_DIM],
                out = self.fc(out_features)  # input [NCHW] = [N x BASE_DIM],
                return out.squeeze()

            def _gradient_penalty(self, real_data, generated_data, labels, device="cpu"):
                self.batch_size = real_data.size()[0]

                # Calculate interpolation
                self.alpha = torch.rand(self.batch_size, 1, 1, 1)
                self.alpha = self.alpha.expand_as(real_data).to(device)
                self.interpolated = self.alpha * real_data + (1 - self.alpha) * generated_data
                #self.interpolated = Variable(self.interpolated.detach(), requires_grad=True).to(device)

                # Calculate probability of interpolated examples
                self.prob_interpolated = self.forward(self.interpolated, labels)
                self.ones_interpolated = torch.ones(self.prob_interpolated.size()).to(device)

                # Calculate gradients of probabilities with respect to examples
                self.gradients = torch_grad(outputs=self.prob_interpolated, inputs=self.interpolated,
                                       grad_outputs=self.ones_interpolated,
                                       create_graph=True, retain_graph=True)[0]

                # Gradients have shape (batch_size, num_channels, img_width, img_height),
                # so flatten to easily take norm per example in batch
                self.gradients = self.gradients.view(self.batch_size, -1)

                # Derivatives of the gradient close to 0 can cause problems because of
                # the square root, so manually calculate norm and add epsilon
                self.gradients_norm = torch.sqrt(torch.sum(self.gradients ** 2, dim=1) + 1e-12)

                # Return gradient penalty
                return 1.0 * ((self.gradients_norm - 1) ** 2).mean()

        class Classifier(nn.Module):
            def __init__(self, ngpu):
                super(Classifier, self).__init__()
                self.ngpu = ngpu
                self.ndf = 64
                self.nc = 3
                self.nemb = 1024
                self.nproj = 32

                self.num_residual = 4
                base_dim = 64

                # First residual block

                self.conv_first = CriticFirstBlock(
                    in_channels=3,
                    out_channels=base_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    spec_norm=True,
                )

                # Make residual block
                for i in range(self.num_residual):

                    if i == 0 or i == 1 or i == 2:
                        resample = 'down'
                    else:
                        resample = None

                    self.add_module('residual_0' + str(i + 1),
                                    ResidualBlock(
                                        in_channels=min(256, base_dim * (2 ** i)),
                                        out_channels=min(256, base_dim * (2 ** (i + 1))),
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        batch_norm=False,
                                        spec_norm=True,
                                        resample=resample,
                                    ))

                    self.fc = spectral_norm(nn.Linear(min(256, base_dim * 2 ** self.num_residual), self.nemb)) #1024-> 256?
                    torch.nn.init.orthogonal_(self.fc.weight)

            def forward(self, x, labels):
                out_critic = self.conv_first(x)

                for i in range(self.num_residual):
                    layer_name = 'residual_0' + str(i + 1)
                    for name, module in self.named_children():
                        if layer_name == name:
                            out_critic = module(out_critic, labels)
                            # input [NCHW] = [N x BASE_DIM x 8 x 8],
                out_features = torch.sum(F.relu(out_critic, inplace=True),
                                         dim=(2, 3))  # output[NCHW] = [N x BASE_DIM],
                out = self.fc(out_features)  # input [NCHW] = [N x BASE_DIM],
                return out.squeeze()

        def save_checkpoint(netD, netG, dir_path, epoch):
            path = os.path.abspath(dir_path)
            if not os.path.exists(path):
                os.makedirs(path)

            torch.save(netD.state_dict(), '{0}/disc_{1}.pth'.format(path, epoch))
            torch.save(netG.state_dict(), '{0}/gen_{1}.pth'.format(path, epoch))

        def smooth_label(tensor, offset):
            return tensor + offset

        def denorm(tensor):
            # std = torch.Tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1)
            std = torch.Tensor([0.5, 0.5, 0.5]).reshape(-1, 1, 1)
            # mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1)
            mean = torch.Tensor([0.5, 0.5, 0.5]).reshape(-1, 1, 1)
            res = torch.clamp(tensor * std + mean, 0, 1)
            return res


        #device = torch.device("cuda:1" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
        device = torch.device('cuda:0')
        # custom weights initialization called on netG and netD
        '''
        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif classname.find('BatchNorm') != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)
        '''
        def weights_init(m):
            classname = m.__class__.__name__
            if type(m) == nn.Conv2d:
                torch.nn.init.orthogonal_(m.weight)
            elif classname.find('Linear') != -1:
                nn.init.orthogonal_(m.weight, gain=2)
            elif classname.find('Conv') != -1:
                nn.init.orthogonal_(m.weight)
            elif classname.find('BatchNorm') != -1:
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)

        netG = Generator(ngpu).to(device)

        # Handle multi-gpu if desired
        #if (device.type == 'cuda') and (ngpu > 1):
        #    netG = nn.DataParallel(netG, list(range(ngpu)))

        # Apply the weight_init function to randomly initialize all weights
        # to mean=0, stdev=0.02
        #etG.apply(weights_init)

        print(netG)

        netD = Discriminator(ngpu).to(device)

        # Handle nulti-gpu if desired
        #if (device.type == 'cuda') and (ngpu > 1):
        #    netD = nn.DataParallel(netD, list(range(ngpu)))

        # Apply the weights_init function to randomly initialize all weights
        # to mean=0, stdev=0.2.
        #netD.apply(weights_init)

        print(netD)

        netC = Classifier(ngpu).to(device)
        print(netC)

        # Initialize BCELoss function
        criterion = nn.BCELoss()

        # Create batch of latent vectors that we will use to visualize
        # the progression of the generator
        # fixed_noise = torch.randn(batch_size, nz, image_size, image_size, device=device)
        # fixed_embed = torch.randn(batch_size, nemb, device=device)

        # Establish convention for real and fake labels during trainig
        real_label = 1.
        fake_label = 0.

        # Setup Adam optimizers for both G and D
        optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
        optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
        optimizerC = optim.Adam(netC.parameters(), lr=0.0005, betas=(0.9, 0.999))

        sample = next(data_iterator)
        print(sample['right_images'][0].shape)
        print(sample['right_embed'][0].shape)
        print(len(sample['txt'][0]))

        # Training
        # Lists to keep track of progress
        img_list = []
        G_losses = []
        D_losses = []
        # iters = 0

        #torch.autograd.set_detect_anomaly(True)
        print("Starting Training Loop...")
        # For each epoch
        #with torch.autograd.set_detect_anomaly(True):
        #with torch.autograd.set_detect_anomaly(True):
        for epoch in range(num_epochs):
            # For each batch in the dataloader
            for i, data in enumerate(dataloader, 0):
                ####################
                # (1) Update D network: maximize log(D(x)) + log(1-D(G(z)))
                ####################
                #if i % 2 ==0:
                netD.zero_grad()

                real_img = data['right_images'].to(device)
                real_embed = data['right_embed'].to(device)
                wrong_img = data['wrong_images'].to(device)
                inter_embed = data['inter_embed'].to(device)

                label = torch.full((real_img.size(0),), real_label, dtype=torch.float, device=device)
                smoothed_real_labels = torch.FloatTensor(smooth_label(label.cpu().numpy(), -0.1)).cuda()
                beta_INT = torch.rand(real_embed.size(0), 1).to(device)
                emb_indices = torch.randperm(real_embed.size(0)).to(device)
                # train with {real image, right text} -> real
                output = netD(real_img, real_embed).view(-1)
                score_r = G_loss(output, smoothed_real_labels)
                #score_r.backward(retain_graph=True)
                D_r = output.mean().item()

                '''
                # train with {real image, wrong text} -> fake
                label.fill_(fake_label)
                beta_INT = 0.5
                output = netD(real_img, real_embed*beta_INT+inter_embed*(1-beta_INT)).view(-1)
                score_w = hinge_loss_fake(output, label)
                score_w.backward()
                D_w = output.mean().item()
                '''

                # train with {fake image, right text} -> fake
                noise = torch.randn(real_img.size(0), nz, dtype=torch.float, device=device)
                fake_images = netG(noise, real_embed*beta_INT+torch.index_select(real_embed, dim=0, index=emb_indices)*(1-beta_INT)) ##
                output = netD(fake_images.detach(), real_embed*beta_INT+torch.index_select(real_embed, dim=0, index=emb_indices)*(1-beta_INT)).view(-1)
                score_f = G_loss(output, label) # hinge loss-> G-loss
                #score_f.backward(retain_graph=True)
                D_f = output.mean().item()

                GP = netD._gradient_penalty(real_data=real_img, generated_data=fake_images, labels=real_embed, device=device)
                l2_D = 0.
                for name, p in netD.named_parameters():
                    if 'bias' in name:
                        l2_D += (torch.square(p)).sum()


                errD = score_r - score_f + l2_D*0.01 + 10.* GP
                errD.backward()

                optimizerD.step()

                try:
                    data_c = next(dataloader_iterator)
                except:
                    dataloader_iterator = iter(dataloader_c)
                    data_c = next(dataloader_iterator)

                real_img_c = data_c['right_images'].to(device)
                real_embed_c = data_c['right_embed'].to(device)

                netC.zero_grad()
                output_c = netC(real_img_c, real_embed_c)
                score_c = L2_loss(output_c, real_embed_c)
                score_c.backward()
                errC = score_c.item()
                optimizerC.step()

                # # Train with all-fake batch
                # # Generate batch of latent vectors
                # noise = torch.randn(real_img.size(0), nz, device=device)
                # wrong_embed = torch.rand(real_img.size(0), nemb, device=device)
                # # Generate batch of latent vectors
                # fake = netG(noise, real_embed)
                #
                # # classify all fake batch with D
                # output = netD(fake.detach(), real_embed).view(-1)
                # errD_fake = criterion(output, label)
                # # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                # errD_fake.backward()
                # D_G_z1 = output.mean().item()
                # errD = errD_real + errD_fake

                # Update D

                ####################
                # (2) Update G network: maximize log(D(g(z)))
                ####################
                netG.zero_grad()
                netD.zero_grad()
                label.fill_(real_label)
                if i % num_train_D == 0:
                    noise = torch.randn(real_img.size(0), nz, dtype=torch.float, device=device)
                    fake_images = netG(noise, real_embed*beta_INT+torch.index_select(real_embed, dim=0, index=emb_indices)*(1-beta_INT))
                    output = netD(fake_images, real_embed).view(-1)
                    output_c = netC(fake_images, real_embed)

                    l2_G = 0.
                    for name, p in netG.named_parameters():
                        if 'bias' in name:
                            l2_G += (torch.square(p)).sum()

                    errG = G_loss(output, label) + 1.0*L2_loss(output_c, real_embed*beta_INT+torch.index_select(real_embed, dim=0, index=emb_indices)*(1-beta_INT)) + l2_G*0.01
                    errG.backward()
                    D_G_z2 = output.mean().item()
                    #loss.backward()
                    optimizerG.step()

                if i % test_interval == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tLoss_C: %.4f\tD(r): %.4f\tD(f): %.4f\tL2_D: %.4f\tL2_G: %.4f'
                          # 'D(G(z)): %.4f / %.4f'
                          % (epoch, num_epochs, i, len(dataloader),
                             errD.item(), errG.item(), errC, D_r, D_f, l2_D.item(), l2_G.item()))

                G_losses.append(errG.item())
                D_losses.append(errD.item())

                del errD

                if i % test_interval == 0 or i == len(dataloader)-1:
                    with torch.no_grad():
                        temp_noise = torch.randn(real_img.size(0), nz, device=device)
                        fake = netG(temp_noise, real_embed).detach().cpu()
                        # img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                        output = denorm(fake.cpu())
                        output_name = os.path.abspath(output_dir) + '/output_epoch_{:0}.jpg'.format(epoch)
                        save_image(output, str(output_name))
                        # fig = plt.figure(figsize=(8, 8))
                        # out = np.transpose(img_list[-1], (1, 2, 0))
                        # plt.imshow(out)
                        # plt.show()
                        # plt.savefig('./results/gen_images_{0}.png'.format(epoch))

                if (epoch+1) % save_model_interval == 0 or epoch == num_epochs-1:
                    save_checkpoint(netD, netG, checkpoint_dir, epoch+1)

                # iters += 1
                torch.cuda.empty_cache()

            plt.figure(figsize=(10, 5))
            plt.title("Generator and Discriminator Loss During Training")
            plt.plot(G_losses, label="G")
            plt.plot(D_losses, label="D")
            plt.xlabel("iterations")
            plt.ylabel("Loss")
            plt.legend()
            # plt.show()
            #plt.close(fig_name)
            fig_name = os.path.abspath(output_dir) + '/plot_epoch_{:0}.jpg'.format(epoch)
            plt.savefig(fig_name)
