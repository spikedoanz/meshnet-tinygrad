# from torch import nn
from tinygrad import Tensor
from tinygrad import nn
import json
#
# class MeshNet(nn.Module):
#     """
#     MeshNet Neural Network
#
#     Arguments:
#         config: config of the neural network
#         bn_before: apply batch normalization before activation function
#         weight_initilization: weight intilization type
#     """
#
#     def __init__(self, config, bn_before=True, 
#         weight_initilization='xavier_uniform'):
#         super(MeshNet, self).__init__()
#         self.model = nn.Sequential()
#         for i, p in enumerate(config):
#             if i != len(config) - 1:
#                 self.model.add_module('conv_{}'.format(i), nn.Conv3d(**p['params']))
#                 if bn_before:
#                     self.model.add_module('bn_{}'.format(i), 
#                         nn.BatchNorm3d(p['params']['out_channels']))
#                 self.model.add_module('relu_{}'.format(i), nn.ReLU(inplace=True))
#                 if not bn_before:
#                     self.model.add_module('bn_{}'.format(i), 
#                         nn.BatchNorm3d(p['params']['out_channels']))
#                 if p['dropout'] > 0:
#                     self.model.add_module('dp_{}'.format(i), 
#                         nn.Dropout3d(p=p['dropout'], inplace=True))
#             else:
#                 self.model.add_module('conv_{}'.format(i), nn.Conv3d(**p['params']))
#
#         # weight initilization
#         weight_init(self.model, weight_initilization)
#
#
#     def forward(self, x):
#         """
#         Forward propagation.
#
#         Arguments:
#             x: input
#         """
#         x = self.model(x)
#         return x
#

class MeshNetTG:
    def __init__(self, config, model_dot_pth):
        with open(config, "r") as f:
            jayson = f.read()
            self.config = json.loads(jayson)
        self.state_dict = nn.from_torch(model_dot_pth)
        for i, p in enumerate(config):
            print(i, p)


if __name__=="__main__":
    model = MeshNetTG("config.json", None)
