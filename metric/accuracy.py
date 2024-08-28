import torch
import torch.nn as nn


class AccuracyCalculator(nn.Module):
    def __init__(self):
        super(AccuracyCalculator, self).__init__()

    def forward(self, predicted, target):
        predicted = predicted > 0.5
        target = target > 0.5

        correct = (predicted == target).float()
        accuracy = correct.sum() / len(target)

        return accuracy
