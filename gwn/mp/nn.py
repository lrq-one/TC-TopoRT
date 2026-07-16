import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch.nn import BatchNorm1d as BN, LayerNorm as LN, Identity

def get_nonlinearity(nonlinearity, return_module=True):
    if nonlinearity == 'relu':
        module = torch.nn.ReLU
        function = F.relu
    elif nonlinearity == 'elu':
        module = torch.nn.ELU
        function = F.elu
    elif nonlinearity == 'gelu':
        module = torch.nn.GELU
        function = F.gelu
    elif nonlinearity == 'id':
        module = torch.nn.Identity
        function = lambda x: x
    elif nonlinearity == 'sigmoid':
        module = torch.nn.Sigmoid
        function = torch.sigmoid 
    elif nonlinearity == 'tanh':
        module = torch.nn.Tanh
        function = torch.tanh
    else:
        raise NotImplementedError('Nonlinearity {} is not currently supported.'.format(nonlinearity))
    
    if return_module:
        return module
    return function

def get_pooling_fn(readout):
    if readout == 'sum':
        return global_add_pool
    elif readout == 'mean':
        return global_mean_pool
    else:
        raise NotImplementedError('Readout {} is not currently supported.'.format(readout))

def get_graph_norm(norm):
    if norm == 'bn':
        return BN
    elif norm == 'ln':
        return LN
    elif norm == 'id':
        return Identity
    else:
        raise ValueError(f'Graph Normalisation {norm} not currently supported')

def pool_complex(xs, data, max_dim, readout_type):
    pooling_fn = get_pooling_fn(readout_type)
    # All complexes have nodes so we can extract the batch size from cochains[0]
    batch_size = data.cochains[0].batch.max() + 1
    
    device = xs[0].device 
    
    # The MP output is of shape [message_passing_dim, batch_size, feature_dim]
    pooled_xs = torch.zeros(max_dim+1, batch_size, xs[0].size(-1), device=device)
    
    for i in range(len(xs)):
        if data.cochains[i].batch is None or xs[i].size(0) == 0:
            continue  
            
        # It's very important that size is supplied.
        pooled_xs[i, :, :] = pooling_fn(xs[i], data.cochains[i].batch, size=batch_size)
        
    return pooled_xs
