# coding: utf-8
import os
import torch

import preprocessing
import utils
from tensorboardX import SummaryWriter

from model import G, D
from test import test_all

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class CGAN:
    def __init__(self, config: dict):
        self.config = config

        self.gen = G().to(device)
        self.dis = D().to(device)

        self.gen_op = torch.optim.Adam(self.gen.parameters(), lr=config['lr'])
        self.dis_op = torch.optim.Adam(self.dis.parameters(), lr=config['lr'])

        self.lda = config['lda']
        self.epsilon = config['epsilon']

    def train_step(self, vis_img, inf_img, vis_y, inf_y, k=2):
        # self.gen.train()

        d_loss_val = 0
        g_loss_val = 0
        fusion = self.gen(inf_img, vis_img) # [32, 1, 120, 120]
        with torch.no_grad():
            fusion_detach = fusion

        # update discriminator 2 times
        for _ in range(k):
            self.dis_op.zero_grad() # reset optimizer grad
            # Discriminator Loss between vis_label and fuse_img
            vis_output = self.dis(vis_y) # [32, 1] D_vis_label, label_shape == fusion_shape != img_shape
            fus_output = self.dis(fusion_detach) # [32, 1] D_fusion
            dis_loss = self.dis_loss_func(vis_output, fus_output)
            d_loss_val += dis_loss.cpu().item()
            dis_loss.backward(retain_graph=True)
            self.dis_op.step()

        # update generator 1 time
        self.gen_op.zero_grad()
        fus_output = self.dis(fusion) # [32, 1] D_fusion
        # Generator Loss, GAN Loss, Content Loss using fuse_img, vis_label, inf_label
        g_loss, v_gan_loss, content_loss = self.gen_loss_func(fus_output, fusion, vis_y, inf_y)
        g_loss_val += g_loss.cpu().item()
        g_loss.backward(retain_graph=False)
        self.gen_op.step()

        return d_loss_val / k, g_loss_val, v_gan_loss, content_loss

    @staticmethod # 静态方法无需实例化
    def dis_loss_func(vis_output, fusion_output):
        x1 = vis_output - torch.Tensor(vis_output.shape).uniform_(0.7, 1.2).to(device)
        x2 = fusion_output - torch.Tensor(fusion_output.shape).uniform_(0, 0.3).to(device)
        return torch.mean(torch.mul(x1,x1)) + \
               torch.mean(torch.mul(x2,x2))

    def gen_loss_func(self, fusion_output, fusion_img, vis_img, inf_img):
        x1 = fusion_output - torch.Tensor(fusion_output.shape).uniform_(0.7, 1.2).to(device)
        x2 = fusion_img - inf_img
        x3 = utils.gradient(fusion_img) - utils.gradient(vis_img)
        gan_loss = torch.mean(torch.mul(x1,x1))
        content_loss = torch.mean(torch.mul(x2,x2)) + \
                       self.epsilon * torch.mean(torch.mul(x3,x3))
        return gan_loss + self.lda * content_loss, gan_loss, self.lda * content_loss


    def train(self):
        if self.config['is_train']:
            data_dir_ir = os.path.join(self.config['data'], 'Train_ir')# data/Train_ir
            data_dir_vi = os.path.join(self.config['data'], 'Train_vi')# data/Train_vi
        else:
            data_dir_ir = os.path.join(self.config['data'], 'Test_ir')# data/Test_ir
            data_dir_vi = os.path.join(self.config['data'], 'Test_vi')# data/Test_vi

        # image_size = 132, label_size = 120, stride = 14
        # get all patches and labels  (37710, 132, 132, 1) (37710, 120, 120, 1)
        train_data_ir, train_label_ir = preprocessing.get_images2(data_dir_ir, self.config['image_size'],
                                                                  self.config['label_size'], self.config['stride'])
        train_data_vi, train_label_vi = preprocessing.get_images2(data_dir_vi, self.config['image_size'],
                                                                  self.config['label_size'], self.config['stride'])
        random_index = torch.randperm(len(train_data_ir)) # a list of random indices
        # shuffle randomly
        train_data_vi = train_data_vi[random_index]
        train_data_ir = train_data_ir[random_index]
        train_label_vi = train_label_vi[random_index]
        train_label_ir = train_label_ir[random_index]
        batch_size = self.config['batch_size']
        print('get patch done')

        if self.config['is_train']:
            with SummaryWriter(self.config['summary_dir']) as writer:
                batch_steps = len(train_data_ir) // batch_size
                epochs = self.config['epoch']
                # for each epoch
                for epoch in range(1, 1 + epochs):
                    d_loss_mean = 0
                    g_loss_mean = 0
                    content_loss_mean = 0
                    # for each step
                    for step in range(1, 1 + batch_steps):
                        print('Training. Epoch:%d/%d Step:%d/%d'%(epoch,epochs,step,batch_steps))
                        start_idx = (step - 1) * batch_size
                        # (32, 132, 132, 1) -> (32, 1, 132, 132)
                        inf_x = train_data_ir[start_idx: start_idx + batch_size].transpose([0, 3, 1, 2])
                        inf_y = train_label_ir[start_idx: start_idx + batch_size].transpose([0, 3, 1, 2])
                        vis_x = train_data_vi[start_idx: start_idx + batch_size].transpose([0, 3, 1, 2])
                        vis_y = train_label_vi[start_idx: start_idx + batch_size].transpose([0, 3, 1, 2])

                        inf_x = torch.tensor(inf_x).float().to(device)
                        inf_y = torch.tensor(inf_y).float().to(device)
                        vis_x = torch.tensor(vis_x).float().to(device)
                        vis_y = torch.tensor(vis_y).float().to(device)

                        d_loss, g_loss, v_gan_loss, content_loss = self.train_step(vis_x, inf_x, vis_y, inf_y, 2)
                        d_loss_mean += d_loss
                        g_loss_mean += g_loss
                        content_loss_mean += content_loss
                        print('Epoch {}/{}, Step {}/{}, gen loss = {:.4f}, v_gan_loss = {:.4f}, '
                              'content_loss {:.4f}, dis loss = {:.4f}'.format(epoch, epochs, step, batch_steps,
                                                                              g_loss, v_gan_loss, content_loss, d_loss))
                    test_all(self.gen, os.path.join(self.config['output'], 'test{}'.format(epoch)))

                    d_loss_mean /= batch_steps
                    g_loss_mean /= batch_steps
                    content_loss_mean /= batch_steps
                    writer.add_scalar('scalar/gen_loss', g_loss_mean, epoch)
                    writer.add_scalar('scalar/dis_loss', d_loss_mean, epoch)
                    writer.add_scalar('scalar/content_loss', content_loss_mean, epoch)

                    # for name, param in self.gen.named_parameters():
                    #     if 'bn' not in name:
                    #         writer.add_histogram('gen/'+name, param, epoch)
                    #
                    # for name, param in self.dis.named_parameters():
                    #     if 'bn' not in name:
                    #         writer.add_histogram('dis/'+name, param, epoch)
            print('Saving model......')
            torch.save(self.gen.state_dict(), '%s/final_generator.pth' % (self.config['output']))
            print("Training Finished, Total EPOCH = %d" % self.config['epoch'])
