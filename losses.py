import torch.optim as optim
import torch
import sys
from models.vid2vid_model_G import Vid2VidModelG
from models.

"""
    This method should return a numpy array of images with shape (batch_size, load_height, load_width, channels). Default load width should be 1024.
"""
def load_target_images(load_width=1024):
    pass

"""
    This method loads a pretrained model, using a similar command-line args scheme as test.py.
"""
def load_model(path):
    model = Vid2VidModelG() 
    model.load_network(
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
