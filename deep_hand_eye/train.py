import os
import argparse

import numpy as np
import random
from tqdm import tqdm

import torch
import torch.utils.tensorboard as tb
from torchvision.models import resnet34

import deep_hand_eye.utils as dhe_utils


config = dhe_utils.AttrDict()

config.max_epochs = 100




def seed_everything(seed: int):
    """From https://pytorch-lightning.readthedocs.io/en/latest/_modules/pytorch_lightning/utilities/seed.html#seed_everything"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Trainer(object):

    def __init__(self, config) -> None:
        self.model = None
        self.train_dataloader = None
        self.config = config

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tb_writer = tb.writer.SummaryWriter()


        # Define the model

    def train(self, max_epochs):
        self.model.train()

        for epoch in tqdm(range(max_epochs), desc='epoch', total=max_epochs):
            pass



def main():
    seed_everything(0)
    
    trainer = Trainer()

