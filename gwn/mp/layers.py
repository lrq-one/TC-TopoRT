import torch
import torch.nn as nn

from typing import Any, Callable, Optional
from torch import Tensor
from mp.cell_mp import CochainMessagePassing, CochainMessagePassingParams
from torch_geometric.nn.inits import reset
from torch.nn import Linear, Sequential, BatchNorm1d as BN, Identity
from mp.complex import Cochain
from torch_scatter import scatter
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from abc import ABC, abstractmethod

import torch.nn.functional as F
class IntraCellularAttention(nn.Module):
    def __init__(self, hidden_dim, num_sources, dropout=0.0): # [新增] dropout 参数默认为 0.1
        super(IntraCellularAttention, self).__init__()
        self.num_sources = num_sources
        
        # 评分网络
        self.score_net = nn.Sequential(
            nn.Linear(hidden_dim * num_sources, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_sources)
        )
        
        # 融合网络
        self.fusion_net = nn.Sequential(
            nn.Linear(hidden_dim * num_sources, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )
        
        # [终极武器]：注意力机制的专属 Dropout
        self.attn_dropout = nn.Dropout(dropout) 

    def forward(self, source_list):
        """
        source_list: 包含各个方向消息的列表，每个 tensor shape 为 [N, hidden_dim]
        """
        # 1. 自动检测全零特征并生成 Mask
        masks = []
        for feat in source_list:
            # mask 为 1 表示有信号，为 0 表示是填充的全零占位符
            mask = (feat.abs().sum(dim=-1, keepdim=True) > 1e-6).float()
            masks.append(mask)
        combined_mask = torch.cat(masks, dim=-1) # [N, num_sources]

        # 2. 计算原始 logits (未归一化的得分)
        concat_feat = torch.cat(source_list, dim=-1)
        logits = self.score_net(concat_feat) # [N, num_sources]
        
        # 3. 强制屏蔽：将全零特征对应的 logits 设为极小值 (-1e9)
        combined_mask[:, 0] = 1.0 
        logits = logits.masked_fill(combined_mask == 0, -1e9)
        
        # ==========================================
        # 4. [修复]：Attention Dropout 施加在 logits 上（softmax 之前）
        # 这样 softmax 后权重之和仍然为 1，训练/推理行为一致。
        # 之前放在 softmax 之后会破坏归一化，导致训练不稳定！
        # ==========================================
        logits = self.attn_dropout(logits)
        
        # 5. 计算注意力分数
        attn_scores = F.softmax(logits, dim=-1) # [N, num_sources]
        
        # 6. 动态加权
        weighted_sources = []
        for i, feat in enumerate(source_list):
            score = attn_scores[:, i].unsqueeze(1) # [N, 1]
            weighted_sources.append(feat * score)
            
        # 7. 融合输出
        out = self.fusion_net(torch.cat(weighted_sources, dim=-1))
        return out
class DummyCochainMessagePassing(CochainMessagePassing):
    """This is a dummy parameter-free message passing model used for testing."""
    def __init__(self, up_msg_size, down_msg_size, boundary_msg_size=None,
                 use_boundary_msg=False, use_down_msg=True):
        super(DummyCochainMessagePassing, self).__init__(up_msg_size, down_msg_size,
                                                       boundary_msg_size=boundary_msg_size,
                                                       use_boundary_msg=use_boundary_msg,
                                                       use_down_msg=use_down_msg)

    def message_up(self, up_x_j: Tensor, up_attr: Tensor) -> Tensor:
        # (num_up_adj, x_feature_dim) + (num_up_adj, up_feat_dim)
        # We assume the feature dim is the same across al levels
        return up_x_j + up_attr

    def message_down(self, down_x_j: Tensor, down_attr: Tensor) -> Tensor:
        # (num_down_adj, x_feature_dim) + (num_down_adj, down_feat_dim)
        # We assume the feature dim is the same across al levels
        return down_x_j + down_attr

    def forward(self, cochain: CochainMessagePassingParams):
        up_out, down_out, boundary_out = self.propagate(cochain.up_index, cochain.down_index,
                                                    cochain.boundary_index, x=cochain.x,
                                                    up_attr=cochain.kwargs['up_attr'],
                                                    down_attr=cochain.kwargs['down_attr'],
                                                    boundary_attr=cochain.kwargs['boundary_attr'])
        # down or boundary will be zero if one of them is not used.
        return cochain.x + up_out + down_out + boundary_out


class DummyCellularMessagePassing(torch.nn.Module):
    def __init__(self, input_dim=1, max_dim: int = 2, use_boundary_msg=False, use_down_msg=True):
        super(DummyCellularMessagePassing, self).__init__()
        self.max_dim = max_dim
        self.mp_levels = torch.nn.ModuleList()
        for dim in range(max_dim+1):
            mp = DummyCochainMessagePassing(input_dim, input_dim, boundary_msg_size=input_dim,
                                          use_boundary_msg=use_boundary_msg, use_down_msg=use_down_msg)
            self.mp_levels.append(mp)
    
    def forward(self, *cochain_params: CochainMessagePassingParams):
        assert len(cochain_params) <= self.max_dim+1

        out = []
        for dim in range(len(cochain_params)):
            out.append(self.mp_levels[dim].forward(cochain_params[dim]))
        return out


class CINCochainConv(CochainMessagePassing):
    """This is a dummy parameter-free message passing model used for testing."""
    def __init__(self, up_msg_size: int, down_msg_size: int,
                 msg_up_nn: Callable, msg_down_nn: Callable, update_nn: Callable,
                 eps: float = 0., train_eps: bool = False):
        super(CINCochainConv, self).__init__(up_msg_size, down_msg_size, use_boundary_msg=False)
        self.msg_up_nn = msg_up_nn
        self.msg_down_nn = msg_down_nn
        self.update_nn = update_nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def forward(self, cochain: CochainMessagePassingParams):
        out_up, out_down, _ = self.propagate(cochain.up_index, cochain.down_index,
                                             None, x=cochain.x,
                                             up_attr=cochain.kwargs['up_attr'],
                                             down_attr=cochain.kwargs['down_attr'])

        out_up += (1 + self.eps) * cochain.x
        out_down += (1 + self.eps) * cochain.x
        return self.update_nn(out_up + out_down)

    def reset_parameters(self):
        reset(self.msg_up_nn)
        reset(self.msg_down_nn)
        reset(self.update_nn)
        self.eps.data.fill_(self.initial_eps)

    def message_up(self, up_x_j: Tensor, up_attr: Tensor) -> Tensor:
        if up_attr is not None:
            x = torch.cat([up_x_j, up_attr], dim=-1)
            return self.msg_up_nn(x)
        else:
            return self.msg_up_nn(up_x_j)

    def message_down(self, down_x_j: Tensor, down_attr: Tensor) -> Tensor:
        x = torch.cat([down_x_j, down_attr], dim=-1)
        return self.msg_down_nn(x)

"""
原理：它利用了单纯复形（Cell Complex）中的“上邻接”（Upper Adjacency）和“下邻接”（Lower Adjacency）来进行消息传递。
输入：cochain_params（包含节点、边、环的特征和邻接矩阵）。
输出：更新后的特征向量列表 [v_out, e_out, c_out]
"""
class CINConv(torch.nn.Module): #这是基础版的 CWN 卷积层。
    def __init__(self, up_msg_size: int, down_msg_size: int,
                 msg_up_nn: Callable, msg_down_nn: Callable, update_nn: Callable,
                 eps: float = 0., train_eps: bool = False, max_dim: int = 2):
        super(CINConv, self).__init__()
        self.max_dim = max_dim
        self.mp_levels = torch.nn.ModuleList()
        for dim in range(max_dim+1):
            mp = CINCochainConv(up_msg_size, down_msg_size,
                              msg_up_nn, msg_down_nn, update_nn, eps, train_eps)
            self.mp_levels.append(mp)

    def forward(self, *cochain_params: CochainMessagePassingParams):
        assert len(cochain_params) <= self.max_dim+1

        out = []
        for dim in range(len(cochain_params)):
            out.append(self.mp_levels[dim].forward(cochain_params[dim]))
        return out


class EdgeCINConv(torch.nn.Module):
    """
    CIN convolutional layer which performs cochain message passing only
    _up to_ 1-dimensional cells (edges).
    """
    def __init__(self, up_msg_size: int, down_msg_size: int,
                 v_msg_up_nn: Callable, e_msg_down_nn: Callable, e_msg_up_nn: Callable,
                 v_update_nn: Callable, e_update_nn: Callable, eps: float = 0., train_eps=False):
        super(EdgeCINConv, self).__init__()
        self.max_dim = 1
        self.mp_levels = torch.nn.ModuleList()

        v_mp = CINCochainConv(up_msg_size, down_msg_size,
                            v_msg_up_nn, lambda *args: None, v_update_nn, eps, train_eps)
        e_mp = CINCochainConv(up_msg_size, down_msg_size,
                            e_msg_up_nn, e_msg_down_nn, e_update_nn, eps, train_eps)
        self.mp_levels.extend([v_mp, e_mp])

    def forward(self, *cochain_params: CochainMessagePassingParams):
        assert len(cochain_params) <= self.max_dim+1

        out = []
        for dim in range(len(cochain_params)):
            out.append(self.mp_levels[dim].forward(cochain_params[dim]))
        return out


class SparseCINCochainConv(CochainMessagePassing):
    """This is a CIN Cochain layer that operates of boundaries and upper adjacent cells."""
    def __init__(self, dim: int,
                 up_msg_size: int,
                 down_msg_size: int,
                 boundary_msg_size: Optional[int],
                 msg_up_nn: Callable,
                 msg_boundaries_nn: Callable,
                 update_up_nn: Callable,
                 update_boundaries_nn: Callable,
                 combine_nn: Callable,
                 eps: float = 0.,
                 train_eps: bool = False):
        super(SparseCINCochainConv, self).__init__(up_msg_size, down_msg_size, boundary_msg_size=boundary_msg_size,
                                                 use_down_msg=False)
        self.dim = dim
        self.msg_up_nn = msg_up_nn
        self.msg_boundaries_nn = msg_boundaries_nn
        self.update_up_nn = update_up_nn
        self.update_boundaries_nn = update_boundaries_nn
        self.combine_nn = combine_nn
        self.initial_eps = eps
        if train_eps:
            self.eps1 = torch.nn.Parameter(torch.Tensor([eps]))
            self.eps2 = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps1', torch.Tensor([eps]))
            self.register_buffer('eps2', torch.Tensor([eps]))
        self.reset_parameters()

    def forward(self, cochain: CochainMessagePassingParams):
        out_up, _, out_boundaries = self.propagate(cochain.up_index, cochain.down_index,
                                              cochain.boundary_index, x=cochain.x,
                                              up_attr=cochain.kwargs['up_attr'],
                                              boundary_attr=cochain.kwargs['boundary_attr'])

        # As in GIN, we can learn an injective update function for each multi-set
        out_up += (1 + self.eps1) * cochain.x
        out_boundaries += (1 + self.eps2) * cochain.x
        out_up = self.update_up_nn(out_up)
        out_boundaries = self.update_boundaries_nn(out_boundaries)

        # We need to combine the two such that the output is injective
        # Because the cross product of countable spaces is countable, then such a function exists.
        # And we can learn it with another MLP.
        return self.combine_nn(torch.cat([out_up, out_boundaries], dim=-1))

    def reset_parameters(self):
        reset(self.msg_up_nn)
        reset(self.msg_boundaries_nn)
        reset(self.update_up_nn)
        reset(self.update_boundaries_nn)
        reset(self.combine_nn)
        self.eps1.data.fill_(self.initial_eps)
        self.eps2.data.fill_(self.initial_eps)

    def message_up(self, up_x_j: Tensor, up_attr: Tensor) -> Tensor:
        return self.msg_up_nn((up_x_j, up_attr))
    
    def message_boundary(self, boundary_x_j: Tensor) -> Tensor:
        return self.msg_boundaries_nn(boundary_x_j)
    
class CINppCochainConv(SparseCINCochainConv):
    """CINppCochainConv
    """
    def __init__(self, dim: int, up_msg_size: int, down_msg_size: int, boundary_msg_size: int, 
                 msg_up_nn: Callable[..., Any], msg_boundaries_nn: Callable[..., Any], msg_down_nn: Callable[..., Any], 
                 update_up_nn: Callable[..., Any], update_boundaries_nn: Callable[..., Any],  update_down_nn: Callable[..., Any], 
                 combine_nn: Callable[..., Any], eps: float = 0, train_eps: bool = False):
        super(CINppCochainConv, self).__init__(dim, up_msg_size, down_msg_size, boundary_msg_size, 
                                               msg_up_nn, msg_boundaries_nn,
                                               update_up_nn, update_boundaries_nn,
                                               combine_nn, eps, train_eps)
        
        self.msg_down_nn = msg_down_nn
        self.update_down_nn = update_down_nn
        if train_eps:
            self.eps3 = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps3', torch.Tensor([eps]))
        
        reset(self.msg_down_nn)
        reset(self.update_down_nn)
        self.eps3.data.fill_(self.initial_eps)


    def message_down(self, down_x_j: Tensor, down_attr: Tensor) -> Tensor:
        return self.msg_down_nn((down_x_j, down_attr))
    

    def forward(self, cochain: CochainMessagePassingParams):
        # 1. 传播：得到纯粹的邻居聚合信息
        out_up, out_down, out_boundaries = self.propagate(
            cochain.up_index, cochain.down_index,
            cochain.boundary_index, x=cochain.x,
            up_attr=cochain.kwargs['up_attr'],
            down_attr=cochain.kwargs.get('down_attr', None),
            boundary_attr=cochain.kwargs['boundary_attr']
        )

        # [二次核心修复]：提取纯净消息掩码！
        # 如果某个方向不存在（如原子的 down），propagate 返回的全是 0。
        # 必须在这里记录掩码，因为紧接着后面的 update_nn 包含 Bias 和 BatchNorm，会把全 0 向量变成非 0！
        # 只有强行乘回 0，你刚刚在 IntraCellularAttention 里写的那些动态掩码逻辑才可能真正生效！
        mask_up = (out_up.abs().sum(dim=-1, keepdim=True) > 1e-6).float()
        mask_down = (out_down.abs().sum(dim=-1, keepdim=True) > 1e-6).float()
        mask_bounds = (out_boundaries.abs().sum(dim=-1, keepdim=True) > 1e-6).float()

        # 2. 映射：对各个方向的纯邻居消息进行非线性变换
        out_up = self.update_up_nn(out_up) * mask_up
        out_down = self.update_down_nn(out_down) * mask_down
        out_boundaries = self.update_boundaries_nn(out_boundaries) * mask_bounds

        # 3. 统一融合：无论什么维度，都让注意力模块去平衡“自身”与“三个方向的邻居”
        # 即使某个方向没有消息（如原子没有 Down 方向），propagate 也会返回零张量，注意力会自动处理
        sources = [cochain.x, out_up, out_down, out_boundaries]
        out = self.combine_nn(sources)

        # 4. 全局残差：只在这里保留一个干净的特征跳连
        return out + cochain.x

class Catter(torch.nn.Module):
    def __init__(self):
        super(Catter, self).__init__()

    def forward(self, x):
        return torch.cat(x, dim=-1)
    
"""
改进版。为了计算效率，它只考虑了特定的消息传递路径（例如：只从边界和上层邻居接收消息），使得计算更稀疏、更快速。
适用场景：大规模分子图或需要高效计算的场景。
"""   
class SparseCINConv(torch.nn.Module):
    """A cellular version of GIN which performs message passing from  cellular upper
    neighbors and boundaries, but not from lower neighbors (hence why "Sparse")
    """

    # TODO: Refactor the way we pass networks externally to allow for different networks per dim.
    def __init__(self, up_msg_size: int, down_msg_size: int, boundary_msg_size: Optional[int],
                 passed_msg_up_nn: Optional[Callable], passed_msg_boundaries_nn: Optional[Callable],
                 passed_update_up_nn: Optional[Callable],
                 passed_update_boundaries_nn: Optional[Callable],
                 eps: float = 0., train_eps: bool = False, max_dim: int = 2,
                 graph_norm=BN, use_coboundaries=False, **kwargs):
        super(SparseCINConv, self).__init__()
        self.max_dim = max_dim
        self.mp_levels = torch.nn.ModuleList()
        for dim in range(max_dim+1):
            msg_up_nn = passed_msg_up_nn
            if msg_up_nn is None:
                if use_coboundaries:
                    msg_up_nn = Sequential(
                            Catter(),
                            Linear(kwargs['layer_dim'] * 2, kwargs['layer_dim']),
                            kwargs['act_module']())
                else:
                    msg_up_nn = lambda xs: xs[0]

            msg_boundaries_nn = passed_msg_boundaries_nn
            if msg_boundaries_nn is None:
                msg_boundaries_nn = lambda x: x

            update_up_nn = passed_update_up_nn
            if update_up_nn is None:
                update_up_nn = Sequential(
                    Linear(kwargs['layer_dim'], kwargs['hidden']),
                    graph_norm(kwargs['hidden']),
                    kwargs['act_module'](),
                    Linear(kwargs['hidden'], kwargs['hidden']),
                    graph_norm(kwargs['hidden']),
                    kwargs['act_module']()
                )

            update_boundaries_nn = passed_update_boundaries_nn
            if update_boundaries_nn is None:
                update_boundaries_nn = Sequential(
                    Linear(kwargs['layer_dim'], kwargs['hidden']),
                    graph_norm(kwargs['hidden']),
                    kwargs['act_module'](),
                    Linear(kwargs['hidden'], kwargs['hidden']),
                    graph_norm(kwargs['hidden']),
                    kwargs['act_module']()
                )
            combine_nn = Sequential(
                Linear(kwargs['hidden']*2, kwargs['hidden']),
                graph_norm(kwargs['hidden']),
                kwargs['act_module']())

            mp = SparseCINCochainConv(dim, up_msg_size, down_msg_size, boundary_msg_size=boundary_msg_size,
                msg_up_nn=msg_up_nn, msg_boundaries_nn=msg_boundaries_nn, update_up_nn=update_up_nn,
                update_boundaries_nn=update_boundaries_nn, combine_nn=combine_nn, eps=eps,
                train_eps=train_eps)
            self.mp_levels.append(mp)

    def forward(self, *cochain_params: CochainMessagePassingParams, start_to_process=0):
        assert len(cochain_params) <= self.max_dim+1

        out = []
        for dim in range(len(cochain_params)):
            if dim < start_to_process:
                out.append(cochain_params[dim].x)
            else:
                out.append(self.mp_levels[dim].forward(cochain_params[dim]))
        return out

class CINppConv(torch.nn.Module): # <--- [修改]：不再继承 SparseCINConv，直接继承 nn.Module
    """
    增强版。考虑了 Up, Down, Boundary 全方向消息传递，表达能力最强。
    """
    def __init__(self, up_msg_size: int, down_msg_size: int, boundary_msg_size: Optional[int],
                 passed_msg_up_nn: Optional[Callable], passed_msg_down_nn: Optional[Callable],
                 passed_msg_boundaries_nn: Optional[Callable],
                 passed_update_up_nn: Optional[Callable],
                 passed_update_down_nn: Optional[Callable],
                 passed_update_boundaries_nn: Optional[Callable],
                 eps: float = 0., train_eps: bool = False, max_dim: int = 2,
                 graph_norm=BN, use_coboundaries=False, **kwargs):
        
        # === [核心修复]：直接初始化自身，避免调用冗余的父类初始化 ===
        super(CINppConv, self).__init__()
        self.max_dim = max_dim
        self.mp_levels = torch.nn.ModuleList()
        
        for dim in range(max_dim+1):
            msg_up_nn = passed_msg_up_nn
            if msg_up_nn is None:
                if use_coboundaries:
                    msg_up_nn = Sequential(
                            Catter(),
                            Linear(kwargs['layer_dim'] * 2, kwargs['layer_dim']),
                            kwargs['act_module']())
                else:
                    msg_up_nn = lambda xs: xs[0]

            msg_down_nn = passed_msg_down_nn
            if msg_down_nn is None:
                if use_coboundaries:
                    msg_down_nn = Sequential(
                            Catter(),
                            Linear(kwargs['layer_dim'] * 2, kwargs['layer_dim']),
                            kwargs['act_module']())
                else:
                    msg_down_nn = lambda xs: xs[0]
                    
            msg_boundaries_nn = passed_msg_boundaries_nn
            if msg_boundaries_nn is None:
                msg_boundaries_nn = lambda x: x

            update_up_nn = passed_update_up_nn
            if update_up_nn is None:
                update_up_nn = Sequential(
                    Linear(kwargs['layer_dim'], kwargs['hidden']),
                    graph_norm(kwargs['hidden']),
                    kwargs['act_module'](),
                    Linear(kwargs['hidden'], kwargs['hidden']),
                    graph_norm(kwargs['hidden']),
                    kwargs['act_module']()
                )
                
            update_down_nn = passed_update_down_nn
            if update_down_nn is None:
                update_down_nn = Sequential(
                    Linear(kwargs['layer_dim'], kwargs['hidden']),
                    graph_norm(kwargs['hidden']),
                    kwargs['act_module'](),
                    Linear(kwargs['hidden'], kwargs['hidden']),
                    graph_norm(kwargs['hidden']),
                    kwargs['act_module']()
                )

            update_boundaries_nn = passed_update_boundaries_nn
            if update_boundaries_nn is None:
                update_boundaries_nn = Sequential(
                    Linear(kwargs['layer_dim'], kwargs['hidden']),
                    graph_norm(kwargs['hidden']),
                    kwargs['act_module'](),
                    Linear(kwargs['hidden'], kwargs['hidden']),
                    graph_norm(kwargs['hidden']),
                    kwargs['act_module']()
                )
            # combine_nn = Sequential(
            #     Linear(kwargs['hidden']*3, kwargs['hidden']),
            #     graph_norm(kwargs['hidden']),
            #     kwargs['act_module']())
            combine_nn = IntraCellularAttention(hidden_dim=kwargs['hidden'], num_sources=4)
            mp = CINppCochainConv(dim, up_msg_size, down_msg_size, boundary_msg_size=boundary_msg_size,
                msg_up_nn=msg_up_nn, msg_down_nn=msg_down_nn, msg_boundaries_nn=msg_boundaries_nn, update_up_nn=update_up_nn,
                update_down_nn=update_down_nn, update_boundaries_nn=update_boundaries_nn, combine_nn=combine_nn, eps=eps,
                train_eps=train_eps)
            self.mp_levels.append(mp)

    # === [核心修复]：补上原本依赖父类的 forward 函数 ===
    def forward(self, *cochain_params: CochainMessagePassingParams, start_to_process=0):
        assert len(cochain_params) <= self.max_dim+1

        out = []
        for dim in range(len(cochain_params)):
            if dim < start_to_process:
                out.append(cochain_params[dim].x)
            else:
                out.append(self.mp_levels[dim].forward(cochain_params[dim]))
        return out


class OrientedConv(CochainMessagePassing):
    def __init__(self, dim: int, up_msg_size: int, down_msg_size: int,
                 update_up_nn: Optional[Callable], update_down_nn: Optional[Callable],
                 update_nn: Optional[Callable], act_fn, orient=True):
        super(OrientedConv, self).__init__(up_msg_size, down_msg_size, use_boundary_msg=False)
        self.dim = dim
        self.update_up_nn = update_up_nn
        self.update_down_nn = update_down_nn
        self.update_nn = update_nn
        self.act_fn = act_fn
        self.orient = orient

    def forward(self, cochain: Cochain):
        assert len(cochain.upper_orient) == cochain.upper_index.size(1)
        assert len(cochain.lower_orient) == cochain.lower_index.size(1)
        assert cochain.upper_index.max() < len(cochain.x)
        assert cochain.lower_index.max() < len(cochain.x)

        out_up, out_down, _ = self.propagate(cochain.upper_index, cochain.lower_index, None, x=cochain.x,
            up_attr=cochain.upper_orient.view(-1, 1), down_attr=cochain.lower_orient.view(-1, 1))

        out_up = self.update_up_nn(out_up)
        out_down = self.update_down_nn(out_down)
        x = self.update_nn(cochain.x)
        return self.act_fn(x + out_up + out_down)

    def reset_parameters(self):
        reset(self.update_up_nn)
        reset(self.update_down_nn)
        reset(self.update_nn)

    # TODO: As a temporary hack, we pass the orientation through the up and down attributes.
    def message_up(self, up_x_j: Tensor, up_attr: Tensor) -> Tensor:
        if self.orient:
            return up_x_j * up_attr
        return up_x_j

    def message_down(self, down_x_j: Tensor, down_attr: Tensor) -> Tensor:
        if self.orient:
            return down_x_j * down_attr
        return down_x_j


class InitReduceConv(torch.nn.Module):

    def __init__(self, reduce='add'):
        """

        Args:
            reduce (str): Way to aggregate boundaries. Can be "sum, add, mean, min, max"
        """
        super(InitReduceConv, self).__init__()
        self.reduce = reduce

    def forward(self, boundary_x, boundary_index):
        # === [核心修复]：防御性检查，处理无环分子 ===
        if boundary_index is None or (isinstance(boundary_index, torch.Tensor) and boundary_index.numel() == 0):
            # 如果没有环结构，直接返回一个与特征维度匹配的空张量
            return torch.zeros((0, boundary_x.size(-1)), device=boundary_x.device)
        features = boundary_x.index_select(0, boundary_index[0])
        out_size = boundary_index[1, :].max() + 1
        return scatter(features, boundary_index[1], dim=0, dim_size=out_size, reduce=self.reduce)

    
class AbstractEmbedVEWithReduce(torch.nn.Module, ABC):
    
    def __init__(self,
                 v_embed_layer: Callable,
                 e_embed_layer: Optional[Callable],
                 init_reduce: InitReduceConv):
        """

        Args:
            v_embed_layer: Layer to embed the integer features of the vertices
            e_embed_layer: Layer (potentially None) to embed the integer features of the edges.
            init_reduce: Layer to initialise the 2D cell features and potentially the edge features.
        """
        super(AbstractEmbedVEWithReduce, self).__init__()
        self.v_embed_layer = v_embed_layer
        self.e_embed_layer = e_embed_layer
        self.init_reduce = init_reduce
    
    @abstractmethod
    def _prepare_v_inputs(self, v_params):
        pass
    
    @abstractmethod
    def _prepare_e_inputs(self, e_params):
        pass
    
    def forward(self, *cochain_params: CochainMessagePassingParams):
        assert 1 <= len(cochain_params) <= 3
        v_params = cochain_params[0]
        e_params = cochain_params[1] if len(cochain_params) >= 2 else None
        c_params = cochain_params[2] if len(cochain_params) == 3 else None

        vx = self.v_embed_layer(self._prepare_v_inputs(v_params))
        out = [vx]

        if e_params is None:
           assert c_params is None
           return out

        reduced_ex = self.init_reduce(vx, e_params.boundary_index)
        ex = reduced_ex
        if e_params.x is not None:
            ex = self.e_embed_layer(self._prepare_e_inputs(e_params))
            # The output of this should be the same size as the vertex features.
            assert ex.size(1) == vx.size(1)
        out.append(ex)

        if c_params is not None:
            # We divide by two in case this was obtained from node aggregation.
            # The division should not do any harm if this is an aggregation of learned embeddings.
            cx = self.init_reduce(reduced_ex, c_params.boundary_index) / 2.
            out.append(cx)

        return out
    
    def reset_parameters(self):
        reset(self.v_embed_layer)
        reset(self.e_embed_layer)

    
class EmbedVEWithReduce(AbstractEmbedVEWithReduce): #将原始数据（Raw Data）转换为可训练的向量（Embeddings），是模型的“入口”。
    """
    用于通用的图数据。将节点特征（通常是整数索引）通过 torch.nn.Embedding 转化为向量。
    重要机制 (init_reduce)：它还会自动初始化**边（Edge）和环（Cell）**的特征。比如，一条边的初始特征可以是它两个端点特征的相加；一个环的特征可以是围成它的边的特征相加。
    """
    def __init__(self,
                 v_embed_layer: torch.nn.Embedding,
                 e_embed_layer: Optional[torch.nn.Embedding],
                 init_reduce: InitReduceConv):
        super(EmbedVEWithReduce, self).__init__(v_embed_layer, e_embed_layer, init_reduce)
        
    def _prepare_v_inputs(self, v_params):
        assert v_params.x is not None
        assert v_params.x.dim() == 2
        assert v_params.x.size(1) == 1
        # The embedding layer expects integers so we convert the tensor to int.
        return v_params.x.squeeze(1).to(dtype=torch.long)
    
    def _prepare_e_inputs(self, e_params):
        assert self.e_embed_layer is not None
        assert e_params.x.dim() == 2
        assert e_params.x.size(1) == 1
        # The embedding layer expects integers so we convert the tensor to int.
        return e_params.x.squeeze(1).to(dtype=torch.long)

"""
OGBEmbedVEWithReduce:
这就是你需要用的！
它是专门为 OGB 分子数据集设计的。
它使用了 OGB 官方提供的 AtomEncoder 和 BondEncoder，能很好地处理原子的化学属性（原子序数、手性等）和化学键属性。
在双塔模型中的作用：这将是你 CWN 分支的第一层，负责把 SMILES 转换来的图数据变成向量。
总结：
这正是你要找的代码。重点关注 OGBEmbedVEWithReduce（入口）和 CINConv / CINppConv（中间层），利用它们搭建你的拓扑特征提取分支。
from models.cwn import CINConv, OGBEmbedVEWithReduce, InitReduceConv
# 假设上面的代码保存在 models/cwn.py 中

class CWNBranch(torch.nn.Module):
    def __init__(self, hidden_dim, num_layers):
        super().__init__()
        
        # 1. 嵌入层：处理原子和键的特征
        self.embed_layer = OGBEmbedVEWithReduce(
            v_embed_layer=AtomEncoder(hidden_dim),
            e_embed_layer=BondEncoder(hidden_dim),
            init_reduce=InitReduceConv(reduce='add')
        )
        
        # 2. 卷积层：提取拓扑特征
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                CINConv(
                    up_msg_size=hidden_dim, 
                    down_msg_size=hidden_dim,
                    msg_up_nn=..., # 这里需要定义一些 MLP
                    msg_down_nn=...,
                    update_nn=...,
                    hidden=hidden_dim
                )
            )
            
    def forward(self, *cochain_params):
        # 1. 嵌入
        xs = self.embed_layer(*cochain_params)
        
        # 2. 卷积
        for conv in self.convs:
            # 更新 cochain_params 中的 x
            for i in range(len(xs)):
                cochain_params[i].x = xs[i]
            xs = conv(*cochain_params)
            
        # 3. Readout (池化)
        # 通常取图级别的特征（例如第0维节点的平均值）
        graph_feature = global_mean_pool(xs[0], batch_index)
        return graph_feature
"""

class OGBEmbedVEWithReduce(AbstractEmbedVEWithReduce):
    
    def __init__(self,
                 v_embed_layer: AtomEncoder,
                 e_embed_layer: Optional[BondEncoder],
                 init_reduce: InitReduceConv):
        super(OGBEmbedVEWithReduce, self).__init__(v_embed_layer, e_embed_layer, init_reduce)

    def _prepare_v_inputs(self, v_params):
        assert v_params.x is not None
        assert v_params.x.dim() == 2
        # Inputs in ogbg-mol* datasets are already long.
        # This is to test the layer with other datasets.
        # return v_params.x.to(dtype=torch.long)
        return v_params.x
    def _prepare_e_inputs(self, e_params):
        assert self.e_embed_layer is not None
        assert e_params.x.dim() == 2
        # Inputs in ogbg-mol* datasets are already long.
        # This is to test the layer with other datasets.
        return e_params.x

