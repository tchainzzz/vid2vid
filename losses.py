import torch.optim as optim
import torch
import sys
from models.vid2vid_model_G import Vid2VidModelG
from options.train_options import TrainOptions 
from data.data_loader import CreateDataLoader
from models.models import create_model


"""
    This method should return a numpy array of images with shape (batch_size, load_height, load_width, channels). Default load width should be 1024.
"""
def load_target_images(load_width=1024):
    pass

"""
    This method loads a pretrained model, using a similar command-line args scheme as test.py.
"""
def load_model(path):

    # options based on testing command:
    # python test.py --name label2city_1024_g1 --label_nc 35 \
    # --loadSize 1024 --n_scales_spatial 3 --use_instance --fg \ 
    # --n_downsample_G 2 --use_single_G

    opt = TrainOptions().parse(save=False)
    opt.label_nc = 35
    opt.loadSize = 1024
    opt.n_scales_spatial = 3
    opt.use_instance = True
    opt.fg = True
    opt.n_downsample_G = 2
    opt.use_single_G = True
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.dataset_mode = 'temporal'

    print('------------ Options -------------')
    for k, v in sorted(vars(opt).items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')


    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)

    return model

"""
    Using pytorch's loss function API, this calculates some loss metric.
"""
def custom_loss(output, target):
    pass

"""
    The training loop.
"""
def main(path="", niters=10, lr=0.0001):
    targets = load_target_images() 
    model = load_model(path)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(niters):
        epoch_loss = 0
        for img in targets:
            optimizer.zero_grad()
            output = model(input.cuda())
            loss = custom_loss(output, target.cuda())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print("loss, epoch {}/{}: {}".format(epoch + 1, niters))
    

import argparse
if __name__=='__main__':
    psr = argparse.ArgumentParser()
    psr.add_argument("--path", type=str, help="Path of the pretrained model")
    psr.add_argument("--niters", type=int, default=10, help="number of training epochs")
    psr.add_argument("--lr", type=float, default=0.0001, help="learning rate (ADAM optimizer)")

    args = psr.parse_args()
    main(**vars(args))
