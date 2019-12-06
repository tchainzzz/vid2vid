import torch.optim as optim
import torch
from torch.autograd import Variable
import sys
from models.vid2vid_model_G import Vid2VidModelG
from options.train_options import TrainOptions 
from data.data_loader import CreateDataLoader
from models.models import create_model



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
    opt.use_instance = True
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
def custom_loss(output, target):
    pass

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

    for epoch in range(opt.niter):
        epoch_loss = 0
        for i, data in enumerate(dataset):
            _, _, height, width = data['A'].size()
            A = Variable(data['A']).view(1, -1, 1, height, width) # folder A has the segmentation maps
            B = Variable(data['B']).view(1, -1, opt.output_nc, height, width) # folder B has the real images
            inst = Variable(data['inst']).view(1, -1, 1, height, width) # instance maps
            
            # FORWARD PASS: GENERATE IMAGE
            optimizer.zero_grad()
            output = model.inference(A, B, inst)

            loss = custom_loss(output, B)
            assert loss is not None, "Loss function cannot return NoneType!"
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print("loss, epoch {}/{}: {}".format(epoch + 1, opt.niter))

        # Save model
        save_filename = 'fewshot_net_G_{}.pth'.format(epoch)
        save_path = os.path.join("./few_shot", save_filename)
        torch.save(model.cpu().state_dict(), PATH)
    

import argparse
if __name__=='__main__':
    main()
