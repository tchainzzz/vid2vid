import os
import torch.optim as optim
import torch
from torch.autograd import Variable
import sys
from models.vid2vid_model_G import Vid2VidModelG
from options.train_options import TrainOptions 
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
import util.util as util
from collections import OrderedDict


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
    opt.use_real_img  = True 
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.dataset_mode = 'temporal'
    opt.isTrain = False 

    print('------------ Options -------------')
    for k, v in sorted(vars(opt).items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')


"""
    Using pytorch's loss function API, this calculates some loss metric.
"""
def CustomLoss(output, target):
    return output - target


"""
    The training loop.
"""
def main():

    opt = TrainOptions().parse(save=False)
    initialize(opt)
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data() # pass into dataroot loc. of our new segmentation maps
    model = create_model(opt)


    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    for param in model.parameters():
        param.requires_grad = True

    for epoch in range(opt.niter):
        epoch_loss = 0
        for i, data in enumerate(dataset):
            _, _, height, width = data['A'].size()
            A = Variable(data['A']).view(1, -1, 1, height, width) # folder A has the segmentation maps
            B = Variable(data['B']).view(1, -1, opt.output_nc, height, width) # folder B has the real images
            
            # FORWARD PASS: GENERATE IMAGE
            optimizer.zero_grad()

            #fake, real = model.inference(A, B, None)
            real_A, real_B, pool_map = model.encode_input(A, B, None)            
            model.is_first_frame = not hasattr(model, 'fake_B_prev') or model.fake_B_prev is None
            if model.is_first_frame:
                model.fake_B_prev = model.generate_first_frame(real_A, real_B, pool_map)      
            
            real_A = model.build_pyr(real_A)            
            model.fake_B_feat = model.flow_feat = model.fake_B_fg_feat = None            
            for s in range(model.n_scales):
                fake_B = model.generate_frame_infer(real_A[model.n_scales-1-s], s)
            fake, real = fake_B, real_A[0][0, -1]


            special_name = data['A_path'][0].split("/")[-1].split("_")[0]            
            util.save_image(util.tensor2im(fake.data[0]), "fake_B_{}_epoch{}.{}".format(special_name, i, 'jpg'))
            loss = CustomLoss(fake, data['B'][:,-1,...].cuda().detach())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print("loss, epoch {}/{}: {}".format(epoch + 1, opt.niter, total_loss))

        # Save model
        save_filename = 'fewshot_net_G_{}.pth'.format(epoch)
        save_path = os.path.join("./few_shot", save_filename)
        torch.save(model.cpu().state_dict(), PATH)
    

import argparse
if __name__=='__main__':
    main()
