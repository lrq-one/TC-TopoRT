import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Linear, Embedding, Sequential, BatchNorm1d as BN
from torch_geometric.nn import AttentionalAggregation
from mp.layers import InitReduceConv, EmbedVEWithReduce, OGBEmbedVEWithReduce, SparseCINConv, CINppConv

try:
    from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
    HAS_OGB = True
except Exception:
    HAS_OGB = False
    class AtomEncoder(torch.nn.Module):
        def __init__(self, emb_dim):
            super().__init__()
            self.embed = Embedding(100, emb_dim)
        def forward(self, x):
            return self.embed(x)

    class BondEncoder(torch.nn.Module):
        def __init__(self, emb_dim):
            super().__init__()
            self.embed = Embedding(10, emb_dim)
        def forward(self, x):
            return self.embed(x)

from mp.complex import ComplexBatch
from mp.nn import pool_complex, get_pooling_fn, get_nonlinearity, get_graph_norm


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        weight = self.fc(x)
        return x * weight

class ContinuousAtomEncoder(nn.Module):
    def __init__(self, in_dim=55, emb_dim=128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim)
        )
        
    def forward(self, x):
        return self.proj(x.float())

class ContinuousBondEncoder(nn.Module):
    def __init__(self, in_dim=21, emb_dim=128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim)
        )
        
    def forward(self, x):
        return self.proj(x.float())


class OGBEmbedSparseCIN(torch.nn.Module):
    def __init__(self, out_size, num_layers, hidden, dropout_rate: float = 0.5, 
                 indropout_rate: float = 0.0, max_dim: int = 2, jump_mode='cat',
                 nonlinearity='gelu', readout='sum', train_eps=False, final_hidden_multiplier: int = 2,
                 readout_dims=(0, 1, 2), final_readout='sum', apply_dropout_before='lin2',
                 init_reduce='sum', embed_edge=False, embed_dim=None, use_coboundaries=False,
                 graph_norm='bn', global_feat_dim=0):
        super(OGBEmbedSparseCIN, self).__init__()

        self.max_dim = max_dim
        self.readout_dims = tuple([dim for dim in readout_dims if dim <= max_dim])
        self.dropout_rate = dropout_rate
        self.in_dropout_rate = indropout_rate
        self.nonlinearity = nonlinearity
        self.readout = readout
        self.jump_mode = jump_mode
        self.final_readout = final_readout

        if embed_dim is None:
            embed_dim = hidden
            
        self.v_embed_init = ContinuousAtomEncoder(in_dim=55, emb_dim=embed_dim)
        self.e_embed_init = ContinuousBondEncoder(in_dim=21, emb_dim=embed_dim)
        self.reduce_init = InitReduceConv(reduce=init_reduce)
        self.init_conv = OGBEmbedVEWithReduce(self.v_embed_init, self.e_embed_init, self.reduce_init)

        self.convs = torch.nn.ModuleList()
        act_module = get_nonlinearity(nonlinearity, return_module=True)
        self.graph_norm = get_graph_norm(graph_norm)
        for i in range(num_layers):
            layer_dim = embed_dim if i == 0 else hidden
            self.convs.append(
                SparseCINConv(up_msg_size=layer_dim, down_msg_size=layer_dim,
                    boundary_msg_size=layer_dim, passed_msg_boundaries_nn=None,
                    passed_msg_up_nn=None, passed_update_up_nn=None,
                    passed_update_boundaries_nn=None, train_eps=train_eps, max_dim=self.max_dim,
                    hidden=hidden, act_module=act_module, layer_dim=layer_dim,
                    graph_norm=self.graph_norm, use_coboundaries=use_coboundaries))
        
        self.total_jk_dim = hidden * (num_layers + 1) if jump_mode == 'cat' else hidden
        concat_dim = (self.max_dim + 1) * self.total_jk_dim  

        self.use_attention_pool = (readout == 'attention')
        if self.use_attention_pool:
            self.att_poolers = torch.nn.ModuleList()
            for _ in range(max_dim + 1):
                gate_nn = Sequential(Linear(self.total_jk_dim, hidden), BN(hidden), act_module(), Linear(hidden, 1))
                self.att_poolers.append(AttentionalAggregation(gate_nn=gate_nn))
                
        self.trans_add = nn.Sequential(
            nn.Linear(concat_dim, concat_dim // 4),  
            nn.GELU(),
            nn.Linear(concat_dim // 4, concat_dim)  
        )
        self.layerNorm_out = nn.LayerNorm(concat_dim)

        self.trans_out = nn.Sequential(
            nn.Dropout(0.15),
            nn.Linear(concat_dim, hidden * 2),   
            nn.LayerNorm(hidden * 2),
            nn.GELU(),
            nn.Linear(hidden * 2, hidden),        
            nn.GELU()
        )
        
        self.regression_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, hidden // 4),
            nn.GELU(),
            nn.Linear(hidden // 4, out_size)
        )   

    def forward(self, data: ComplexBatch, include_partial=False):
        act = get_nonlinearity(self.nonlinearity, return_module=False)
        res = {}

        params = data.get_all_cochain_params(max_dim=self.max_dim, include_down_features=True)
        xs = list(self.init_conv(*params))
        for i in range(len(xs)):
            xs[i] = F.dropout(xs[i], p=self.in_dropout_rate, training=self.training)
        
        jk_list = [[x.clone() for x in xs]] 
        data.set_xs(xs)

        for c, conv in enumerate(self.convs):
            params = data.get_all_cochain_params(max_dim=self.max_dim, include_down_features=True)
            xs = conv(*params)
            for i in range(len(xs)):
                xs[i] = F.dropout(xs[i], p=self.dropout_rate, training=self.training)
            data.set_xs(xs)
            jk_list.append([x.clone() for x in xs])

        num_layers_total = len(jk_list)
        final_cochain_xs = []
        for d in range(self.max_dim + 1):
            layer_features = [jk_list[i][d] for i in range(num_layers_total)]
            final_cochain_xs.append(torch.cat(layer_features, dim=-1))

        if self.use_attention_pool:
            batch_size = data.cochains[0].batch.max() + 1
            pooled_xs = []
            for i in range(len(final_cochain_xs)):
                if data.cochains[i].batch is None or final_cochain_xs[i].size(0) == 0:
                    pooled_xs.append(torch.zeros(batch_size, self.total_jk_dim, device=final_cochain_xs[i].device))
                else:
                    pooled_xs.append(self.att_poolers[i](final_cochain_xs[i], data.cochains[i].batch, dim_size=batch_size))
            x = torch.stack(pooled_xs, dim=0)
        else:
            x = pool_complex(final_cochain_xs, data, self.max_dim, self.readout)

        x = x.transpose(0, 1).flatten(start_dim=1)

        score = torch.sigmoid(self.trans_add(x))
        x = x * score  

        x = self.layerNorm_out(x) 

        x = self.trans_out(x)

        x = self.regression_head(x)

        if include_partial:
            res['out'] = x
            return x, res
        return x

    def reset_parameters(self):
        for conv in self.convs: conv.reset_parameters()
        self.init_conv.reset_parameters()
        
        modules_to_reset = [self.trans_add, self.trans_out, self.regression_head]
        for seq in modules_to_reset:
            for m in seq:
                if hasattr(m, 'reset_parameters'): 
                    m.reset_parameters()

    def __repr__(self):
        return self.__class__.__name__


class OGBEmbedCINpp(OGBEmbedSparseCIN):
    def __init__(self, out_size, num_layers, hidden, dropout_rate: float = 0.5,
                 indropout_rate: float = 0, max_dim: int = 2, jump_mode='cat',
                 nonlinearity='gelu', readout='attention', train_eps=False,
                 final_hidden_multiplier: int = 2, readout_dims=(0, 1, 2),
                 final_readout='sum', apply_dropout_before='lin2', init_reduce='sum',
                 embed_edge=True, embed_dim=None, use_coboundaries=False, graph_norm='bn',
                 global_feat_dim=0):
                 
        super().__init__(out_size, num_layers, hidden, dropout_rate, indropout_rate,
                         max_dim, jump_mode, nonlinearity, readout, train_eps,
                         final_hidden_multiplier, readout_dims, final_readout,
                         apply_dropout_before, init_reduce, embed_edge, embed_dim,
                         use_coboundaries, graph_norm, global_feat_dim=global_feat_dim)
        
        self.convs = torch.nn.ModuleList()
        act_module = get_nonlinearity(nonlinearity, return_module=True)
        
        for i in range(num_layers):
            layer_dim = self.v_embed_init.proj[0].out_features if i == 0 else hidden
            self.convs.append(
                CINppConv(up_msg_size=layer_dim, down_msg_size=layer_dim,
                    boundary_msg_size=layer_dim, passed_msg_boundaries_nn=None,
                    passed_msg_up_nn=None, passed_msg_down_nn=None, passed_update_up_nn=None,
                    passed_update_down_nn=None, passed_update_boundaries_nn=None, train_eps=train_eps,
                    max_dim=self.max_dim, hidden=hidden, act_module=act_module, layer_dim=layer_dim,
                    graph_norm=self.graph_norm, use_coboundaries=use_coboundaries))
