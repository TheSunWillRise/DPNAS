import torch
import numpy as np
import torch.nn as nn


class Architect(nn.Module):
    def __init__(self, model, args):
        super(Architect, self).__init__()
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
                                          lr=args.arch_learning_rate, betas=(0.5, 0.999),
                                          weight_decay=args.arch_weight_decay)
        self.baseline = 0
        self.gamma = args.gamma

    def update_baseline(self, reward):
        self.baseline = self.baseline * self.gamma + reward * (1-self.gamma)


    def step(self, input_valid, target_valid, epoch=0):
        self.optimizer.zero_grad()
        loss, reward, normal_ent, reduce_ent = self.model._loss_arch(
            input_valid, target_valid, self.baseline)
        loss.backward()
        self.optimizer.step()
        self.update_baseline(reward)
        return reward, normal_ent, reduce_ent



