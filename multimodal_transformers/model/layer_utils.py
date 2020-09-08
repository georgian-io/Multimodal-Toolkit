import math
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss


class MLP(nn.Module):
    """mlp can specify number of hidden layers and hidden layer channels"""

    def __init__(self, input_dim, output_dim, act='relu', num_hidden_lyr=2,
                 dropout_prob=0.5, return_layer_outs=False,
                 hidden_channels=None, bn=False):
        super().__init__()
        self.out_dim = output_dim
        self.dropout = nn.Dropout(dropout_prob)
        self.return_layer_outs = return_layer_outs
        if not hidden_channels:
            hidden_channels = [input_dim for _ in range(num_hidden_lyr)]
        elif len(hidden_channels) != num_hidden_lyr:
            raise ValueError(
                "number of hidden layers should be the same as the lengh of hidden_channels")
        self.layer_channels = [input_dim] + hidden_channels + [output_dim]
        self.act_name = act
        self.activation = create_act(act)
        self.layers = nn.ModuleList(list(
            map(self.weight_init, [nn.Linear(self.layer_channels[i], self.layer_channels[i + 1])
                                   for i in range(len(self.layer_channels) - 2)])))
        final_layer = nn.Linear(self.layer_channels[-2], self.layer_channels[-1])
        self.weight_init(final_layer,   activation='linear')
        self.layers.append(final_layer)

        self.bn = bn
        if self.bn:
            self.bn = nn.ModuleList([torch.nn.BatchNorm1d(dim) for dim in self.layer_channels[1:-1]])

    def weight_init(self, m, activation=None):
        if activation is None:
            activation = self.act_name
        torch.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain(activation))
        return m

    def forward(self, x):
        """
        :param x: the input features
        :return: tuple containing output of MLP,
                and list of inputs and outputs at every layer
        """
        layer_inputs = [x]
        for i, layer in enumerate(self.layers):
            input = layer_inputs[-1]
            if layer == self.layers[-1]:
                layer_inputs.append(layer(input))
            else:
                if self.bn:
                    output = self.activation(self.bn[i](layer(input)))
                else:
                    output = self.activation(layer(input))
                layer_inputs.append(self.dropout(output))

        # model.store_layer_output(self, layer_inputs[-1])
        if self.return_layer_outs:
            return layer_inputs[-1], layer_inputs
        else:
            return layer_inputs[-1]


def calc_mlp_dims(input_dim, division=2, output_dim=1):
    dim = input_dim
    dims = []
    while dim > output_dim:
        dim = dim // division
        dims.append(int(dim))
    dims = dims[:-1]
    return dims


def create_act(act, num_parameters=None):
    if act == 'relu':
        return nn.ReLU()
    elif act == 'prelu':
        return nn.PReLU(num_parameters)
    elif act == 'sigmoid':
        return nn.Sigmoid()
    elif act == 'tanh':
        return nn.Tanh()
    elif act == 'linear':
        class Identity(nn.Module):
            def forward(self, x):
                return x

        return Identity()
    else:
        raise ValueError('Unknown activation function {}'.format(act))


def glorot(tensor):
    stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
    if tensor is not None:
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


def hf_loss_func(inputs, classifier, labels, num_labels, class_weights):
    logits = classifier(inputs)
    if type(logits) is tuple:
        logits, layer_outputs = logits[0], logits[1]
    else:  # simple classifier
        layer_outputs = [inputs, logits]
    if labels is not None:
        if num_labels == 1:
            #  We are doing regression
            loss_fct = MSELoss()
            labels = labels.float()
            loss = loss_fct(logits.view(-1), labels.view(-1))
        else:
            loss_fct = CrossEntropyLoss(weight=class_weights)
            labels = labels.long()
            loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
    else:
        return None, logits, layer_outputs

    return loss, logits, layer_outputs