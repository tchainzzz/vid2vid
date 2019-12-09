import os
import torch.optim as optim
import torch
from torch.autograd import Variable
import sys
import models
from models.vid2vid_model_G import Vid2VidModelG
from options.train_options import TrainOptions 
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
import time
from pytorch_msssim import ms_ssim, MS_SSIM
import ssim as pytorch_ssim
import matplotlib.pyplot as plt

"""
    This method loads a pretrained model, using a similar command-line args scheme as test.py.
"""
def initialize(opt):

    # options based on testing command:
    # python test.py --name label2city_1024_g1 --label_nc 35 \
    # --loadSize 1024 --n_scales_spatial 3 --use_instance --fg \ 
    # --n_downsample_G 2 --use_single_G

    opt.label_nc = 35
    opt.loadSize = 1024
    opt.n_scales_spatial = 3
    opt.fg = True
    opt.n_downsample_G = 2
    opt.use_real_img = False
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.dataset_mode = 'temporal'
    opt.isTrain = True 
    opt.continue_train = True
    opt.use_single_G = True 

    print('------------ Options -------------')
    for k, v in sorted(vars(opt).items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')


"""
    Using pytorch's loss function API, this calculates some loss metric.
"""
def CustomLoss(output, target):
    target = target.detach()
    return torch.norm(output - target)

def plot_loss(losses, save_path="plot.png"):
    plt.plot(losses)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(save_path)
    


"""
    The training loop.
"""
def main():

    opt = TrainOptions().parse(save=False)
    initialize(opt)
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data() # pass into dataroot loc. of our new segmentation maps
    tG = opt.n_frames_G

    model,_, _ = create_model(opt)

    # freeze layers
    for small_model in model.children():
        for smaller_layer in small_model.children():
            for even_smaller_layer in smaller_layer.children():
                layers_to_freeze = sum(type(x) == models.networks.ResnetBlock for x in even_smaller_layer.children()) // 3
                resnet_count = 0
                set_trainable = True
                for x in even_smaller_layer.children():
                    if type(x) == models.networks.ResnetBlock: resnet_count += 1
                    for param in x.parameters(): param.requires_grad = set_trainable
                    if resnet_count > layers_to_freeze: set_trainable = False
                    
                
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    loss = pytorch_ssim.SSIM(window_size=11) 
    model = model.to(torch.device("cuda"))
    model.fake_B_prev = None

    
    best_loss = float('inf')
    lambda_raw = 10
    losses = []
    for epoch in range(opt.niter):
        total_loss = 0
        start = time.time()
        n_seq = 0
        for i, data in enumerate(dataset):
            _, _, height, width = data['A'].size()
            A = data['A'].view(1, -1, 1, height, width) # folder A has the segmentation maps
            B = data['B'].view(1, -1, opt.output_nc, height, width).requires_grad_(requires_grad=False) # folder B has the real images
            
            n_img = A.shape[1]
            save_seq = range(6) # save first 6 frames of each 
            n_seq += n_img - tG
            for j in range(n_img - tG):
                A_curr = A[:, j:j+tG, ...]
                B_curr = B[:, j:j+tG,...]
			    # FORWARD PASS: GENERATE IMAGE
                optimizer.zero_grad()
                outputs = model(A_curr, B_curr, None, model.fake_B_prev)
                fake = outputs[0]
                fake_raw = outputs[1]
                real_B = outputs[5][:, 1:]
                # util.save_image(util.tensor2im(real_B), "few_shot/fake_B.jpg") # temporary; to sanity check
                real_B = real_B.detach()
	
                if j in save_seq or epoch==opt.niter-1: util.save_image(util.tensor2im(fake.data[0]), "few_shot/results/fake_B_img{:04d}_epoch_{:04d}.{}".format(j, epoch, 'jpg'))
                output = loss(fake, real_B) 
                output.backward(retain_graph=False)
                optimizer.step()
                total_loss += output.item()
                sys.stdout.write("\r[INFO] current loss on image subsequence {}/{}, epoch={}: {}".format(j+1, n_img - tG, epoch+1, output.item()))
        print("\ntotal loss, epoch {}/{}: {}, {} per iamge - took {:.4f}s".format(epoch + 1, opt.niter, total_loss, total_loss / (n_img - tG), time.time() - start))

        # Save model
        losses.append(total_loss)
        if total_loss < best_loss:
            best_loss = total_loss
            save_path = os.path.join("./few_shot/checkpoints", "latest_net_G0.pth") 
        save_filename = 'fewshot_net_G_{}.pth'.format(epoch)
        save_path = os.path.join("./few_shot/checkpoints", save_filename)
        torch.save(model.state_dict(), save_path)
    plot_loss(losses, save_path="plot_" + str(time.time())+".png")
    

import argparse
if __name__=='__main__':
    main()
