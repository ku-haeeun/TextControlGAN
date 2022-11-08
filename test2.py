from nntplib import NNTP_PORT
from controlGAN import Text2ImageDataset
import os
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision.utils import save_image
from collections import OrderedDict
import h5py
import numpy as np
from PIL import Image
from torch.autograd import grad as torch_grad
from torch.autograd import Variable

import torch.nn.functional as F

import torch.nn.utils.spectral_norm as spectral_norm
from utils import CondBatchNorm2d, init_weights, ResidualBlock, CriticFirstBlock


class Tester(object):
    def __init__(self, model_dir, dataset_path, output_dir, ngpu, num_workers, batch_size, nz, ngf, ndf):

        dataset = Text2ImageDataset(dataset_path,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
                                    split=1)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        # data_iterator = iter(dataloader)
        #print('hello world')
        # plt.show()
        print(len(dataloader))

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
                self.conv_first = spectral_norm(nn.Linear(in_features=int(self.nz / 4), out_features=min(1024,
                                                                                                         base_dim * (
                                                                                                                     2 ** (
                                                                                                                 self.num_residual))) * 4 * 4,
                                                          bias=False))

                torch.nn.init.orthogonal_(self.conv_first.weight)
                # Make residual blocks
                for i in range(self.num_residual):
                    self.add_module('residual_0' + str(i + 1), ResidualBlock(
                        in_channels=min(1024, base_dim * (2 ** (self.num_residual - i))),
                        out_channels=min(1024, base_dim * (2 ** (self.num_residual - i - 1))),
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        resample='up'
                    ))

                # Make high resolution imgs
                # Upsampling
                # Batchnorm On/Off Switch will be needed

                self.output_hr = nn.Sequential(

                    nn.BatchNorm2d(base_dim, affine=False, momentum=None),
                    nn.ReLU(inplace=True),
                    spectral_norm(nn.Conv2d(in_channels=base_dim, out_channels=3,
                                            kernel_size=3, stride=1, padding=1)),
                    nn.Tanh(),

                )
                self.output_hr.apply(init_weights)

            def forward(self, z, emb):
                projected_emb = F.relu(self.projection(emb))
                projected_emb = torch.tanh(self.projection2(projected_emb))
                z_list = [z[:, 0:int(z.size(1) / 4)], z[:, int(z.size(1) / 4):2 * int(z.size(1) / 4)],
                          z[:, 2 * int(z.size(1) / 4):3 * int(z.size(1) / 4)], z[:, 3 * int(z.size(1) / 4):]]
                out_gen = self.conv_first(z_list[0]).reshape(z.size(0), -1, 4, 4)

                for i in range(self.num_residual):
                    layer_name = 'residual_0' + str(i + 1)

                    for name, module in self.named_children():
                        if layer_name == name:
                            if i != self.num_residual - 1:
                                out_gen = module(out_gen, projected_emb * z_list[i + 1])
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
                                        in_channels=min(1024, base_dim * (2 ** i)),
                                        out_channels=min(1024, base_dim * (2 ** (i + 1))),
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        batch_norm=False,
                                        spec_norm=True,
                                        resample=resample,
                                    ))

                    self.projection = spectral_norm(nn.Linear(self.nemb, 128))
                    self.projection2 = spectral_norm(nn.Linear(128, self.nproj, bias=False))

                    self.fc = spectral_norm(nn.Linear(min(1024, base_dim * 2 ** self.num_residual), 1, bias=False))
                    torch.nn.init.orthogonal_(self.fc.weight)

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
                # self.interpolated = Variable(self.interpolated.detach(), requires_grad=True).to(device)

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


        device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
        
        ##~~~~~~~~state_dict~~~~~~~~~
        netG = Generator(ngpu).to(device)
        '''
        loaded_state_dict = torch.load(model_dir+'gen_40.pth')
        new_state_dict = OrderedDict()
        for n,v in loaded_state_dict.items():
            name = n.replace('module','')
            name = n.replace('main','netG')
            #name = name[1:]
            new_state_dict[name] = v
        netG.load_state_dict(new_state_dict)    
        #netG.load_state_dict(torch.load(model_dir+'gen_800.pth'))#,strict=False)
        '''
        netG.load_state_dict(torch.load(model_dir+'gen_100.pth'))
        netG.eval()
        f = open("text.txt", 'w', encoding='utf-8')

        print("Starting Testing Loop...")
        # For each epoch
        for k in range(4):
            for i, data in enumerate(dataloader, 0):

                    with torch.no_grad():
                        real_img = data['right_images'].to(device)
                        real_embed = data['right_embed']
                        txt = data['txt'][0]

                        f.write(str(i) + txt + '\n')

                        with torch.no_grad():
                            temp_noise = torch.randn(64, nz, device=device) * 0.5 
                            fake = netG(temp_noise, torch.Tensor(np.tile(real_embed, (64, 1))).to(device)).detach().cpu()
                            output = denorm(fake.cpu()[0])
                            ''' 
                            코드 변경 => 결과 안 좋아
                            np_output = np.array(output)
                            im = Image.fromarray((np_output.transpose((1,2,0))*128+128).astype(np.uint8))
                            im.save(output_name)
                            '''
                            output_name = os.path.abspath(output_dir) + '/output_text_{0}.jpg'.format(i+k*13035)                 
                            output_name_emb = os.path.abspath(output_dir) + '/output_emb_{0}.npy'.format(i+k*13035) 
                            np.save(output_name_emb, real_embed)
                            save_image(output, str(output_name))
                            
                        
                    print(i+k*len(dataloader), txt)
            
        f.close()
        
