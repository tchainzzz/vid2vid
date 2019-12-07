import os
import torch.optim as optim
import torch
from torch.autograd import Variable
import sys
from models.vid2vid_model_G import Vid2VidModelG
from options.train_options import TrainOptions 
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
import time
from util.graph_viz import make_dot

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
    opt.use_real_img = True 
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.dataset_mode = 'temporal'
    opt.isTrain = True 
    opt.continue_train = True 

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


def gen_image(model, A, B):
	real_A, real_B, pool_map = model.encode_input(A, B, None)            
	model.is_first_frame = not hasattr(model, 'fake_B_prev') or model.fake_B_prev is None
	if model.is_first_frame:
		model.fake_B_prev = model.generate_first_frame(real_A, real_B, pool_map) 
		model.is_first_frame = False

	real_A = model.build_pyr(real_A)            
	model.fake_B_feat = model.flow_feat = model.fake_B_fg_feat = None    
			
	for s in range(model.n_scales):
		fake_B = model.generate_frame_infer(real_A[model.n_scales-1-s], s)
	return fake_B, real_A[0][0, -1]


"""
    The training loop.
"""
def main():

    opt = TrainOptions().parse(save=False)
    initialize(opt)
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data() # pass into dataroot loc. of our new segmentation maps

    model,_, _ = create_model(opt)
    print("---------Layers----------")
    for name, child  in model.named_children():
        print(name)

    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    loss = torch.nn.MSELoss() 
    model = model.to(torch.device("cuda"))
    model.fake_B_prev = None
    for param in model.parameters():
        param.requires_grad = True

    for epoch in range(opt.niter):
        total_loss = 0
        start = time.time()
        for i, data in enumerate(dataset):
            _, _, height, width = data['A'].size()
            A = Variable(data['A']).view(1, -1, 1, height, width) # folder A has the segmentation maps
            B = Variable(data['B']).view(1, -1, opt.output_nc, height, width).requires_grad_(requires_grad=False) # folder B has the real images
   
            # FORWARD PASS: GENERATE IMAGE
            optimizer.zero_grad()

            outputs = model(A, B, None, model.fake_B_prev)
            fake = outputs[0]
            B = B.detach()

            util.save_image(util.tensor2im(fake.data[0]), "few_shot/results/fake_B_img{:04d}_epoch_{:04d}.{}".format(i, epoch, 'jpg'))
            output = loss(fake, data['B'][:,-3:,...].cuda())
            output.backward(retain_graph=False)
            optimizer.step()
            total_loss += output.item()
        print("loss, epoch {}/{}: {} - took {:.4f}s".format(epoch + 1, opt.niter, total_loss, time.time() - start))

        # Save model
        save_filename = 'fewshot_net_G_{}.pth'.format(epoch)
        save_path = os.path.join("./few_shot/checkpoints", save_filename)
        torch.save(model.state_dict(), save_path)
    

import argparse
if __name__=='__main__':
    main()
