import torch


class MLP(torch.nn.Module):
    def __init__(self, name, layer_dims, out_act=None):
        super().__init__()
        self.name = name
        self.layers = torch.nn.ModuleList()
        for i in range(len(layer_dims) - 1):
            self.layers.append(torch.nn.Linear(layer_dims[i], layer_dims[i + 1]))
            if i < len(layer_dims) - 2:
                torch.nn.init.kaiming_uniform_(
                    self.layers[2 * i].weight, nonlinearity="relu"
                )
                self.layers.append(torch.nn.ReLU())
            else:
                torch.nn.init.xavier_uniform_(self.layers[2 * i].weight)
        if out_act == "sigmoid":
            self.layers.append(torch.nn.Sigmoid())

    def forward(self, X):
        for layer in self.layers:
            X = layer(X)
        return X
