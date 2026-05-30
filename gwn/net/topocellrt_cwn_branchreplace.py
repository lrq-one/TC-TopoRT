import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import global_add_pool
from torch_scatter import scatter

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from model_topocellrt import AtomBondAttentionConv
from mp.layers import CINppConv
from mp.nn import get_nonlinearity, get_graph_norm


class CWNEdgeBranch(nn.Module):
    """
    只替换原 DualCellBondConv 的超图支路。

    输入当前层:
        node hidden h
        edge hidden edge_h
        ring hidden ring_h

    输出:
        updated edge_h, updated ring_h

    注意：
        不做 pooling
        不做 graph token
        不做 RT head
        只负责替代 DualCellBondConv 的 edge/higher-order message passing
    """

    def __init__(self, hidden=256, max_dim=2, init_scale=0.10):
        super().__init__()
        self.hidden = hidden
        self.max_dim = max_dim

        act_module = get_nonlinearity("gelu", return_module=True)
        graph_norm = get_graph_norm("bn")

        self.cwn_conv = CINppConv(
            up_msg_size=hidden,
            down_msg_size=hidden,
            boundary_msg_size=hidden,
            passed_msg_boundaries_nn=None,
            passed_msg_up_nn=None,
            passed_msg_down_nn=None,
            passed_update_up_nn=None,
            passed_update_down_nn=None,
            passed_update_boundaries_nn=None,
            train_eps=False,
            max_dim=max_dim,
            hidden=hidden,
            act_module=act_module,
            layer_dim=hidden,
            graph_norm=graph_norm,
            use_coboundaries=True,
        )

        # residual scale 防止一开始把原始 edge/ring 表示冲坏
        self.edge_scale = nn.Parameter(torch.tensor(float(init_scale)))
        self.ring_scale = nn.Parameter(torch.tensor(float(init_scale)))
        self.ring_edge_scale = nn.Parameter(torch.tensor(float(init_scale)))

        # CWN-aware gated residual: edge_h 是否接受 CINppConv 的 edge_delta
        self.edge_gate = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.Sigmoid(),
        )

        # ring_h 是否接受 CINppConv 的 ring_delta
        self.ring_gate = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.Sigmoid(),
        )

        # 显式 ring -> edge 反馈：把 ring-cell 信息 scatter 回属于该 ring 的 bond/edge
        self.ring_to_edge = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )

        self.ring_edge_gate = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.Sigmoid(),
        )

    def forward(self, data, h, edge_h, ring_h=None):
        has_ring = (
            data.dimension >= 2
            and hasattr(data, "cochains")
            and 2 in data.cochains
            and ring_h is not None
        )

        if has_ring:
            xs = [h, edge_h, ring_h]
            max_dim = 2
        else:
            xs = [h, edge_h]
            max_dim = 1

        data.set_xs(xs)

        params = data.get_all_cochain_params(
            max_dim=max_dim,
            include_down_features=True,
            include_boundary_features=True,
        )

        out_xs = list(self.cwn_conv(*params))

        # node 由 AtomBondAttentionConv 更新，这里只取 edge/ring 分支
        # plain branchreplace: 先验证 DualCellBondConv -> CINppConv 是否成立
        edge_delta = out_xs[1]
        edge_out = edge_h + torch.tanh(self.edge_scale) * edge_delta

        if has_ring and len(out_xs) > 2:
            ring_delta = out_xs[2]
            ring_out = ring_h + torch.tanh(self.ring_scale) * ring_delta
        else:
            ring_out = ring_h

        return edge_out, ring_out


class TopoCellRTCWNBranchBlock(nn.Module):
    """
    原始:
        AtomBondAttentionConv + DualCellBondConv

    替换:
        AtomBondAttentionConv + CWNEdgeBranch

    保留逐层 node-edge 交互。
    """

    def __init__(self, hidden=256, heads=8, max_dim=2):
        super().__init__()

        self.conv = AtomBondAttentionConv(
            hidden,
            hidden,
            heads=heads,
            edge_dim=hidden,
            beta=True,
            dropout=0,
            concat=True,
        )
        self.BatchNorm = nn.BatchNorm1d(hidden)

        self.cwn_edge = CWNEdgeBranch(
            hidden=hidden,
            max_dim=max_dim,
            init_scale=0.10,
        )

        # 原师兄 DualCellBondConv 用旧 h 参与 edge 更新。
        # CWN 分支可以更强：允许少量使用 h_gat，但初始化时几乎等价于旧 h。
        self.node_mix_logit = nn.Parameter(torch.tensor(-4.0))

        self.BatchNorm2 = nn.BatchNorm1d(hidden)

    def forward(self, data, h, edge_h, directed_edge_index, directed_edge_attr, ring_h=None):
        # 1. 保留师兄原 node attention 分支
        h_gat = self.conv(
            x=h,
            edge_index=directed_edge_index,
            edge_attr=directed_edge_attr,
        )
        h_gat = F.gelu(self.BatchNorm(h_gat))

        # 2. 只把 DualCellBondConv 换成 CWN edge/ring branch
        edge_h2, ring_h2 = self.cwn_edge(
            data=data,
            h=h,
            edge_h=edge_h,
            ring_h=ring_h,
        )
        edge_h2 = F.gelu(self.BatchNorm2(edge_h2))

        return h_gat, edge_h2, ring_h2


class TopoCellRTCWNBranchReplace(nn.Module):
    """
    最终版：单模型内部 block-level 替换。

    保留师兄:
        AtomBondAttentionConv
        6层逐层交互
        graphline
        trans_graph
        trans_add SE gate
        trans_out
        global_feat gate
        out_lin

    替换师兄:
        DualCellBondConv -> CWNEdgeBranch(CINppConv)
    """

    def __init__(self, emb_dim=256, heads=8, max_dim=2):
        super().__init__()
        self.emb_dim = emb_dim
        self.max_dim = max_dim

        self.layerNorm_out = nn.LayerNorm(emb_dim * 2)

        # gwn 的 SMRTComplexDataset 当前是 55D atom / 21D bond
        self.in_node = nn.Linear(55, emb_dim)
        self.in_edge = nn.Linear(21, emb_dim)

        self.conv1 = TopoCellRTCWNBranchBlock(emb_dim, heads=heads, max_dim=max_dim)
        self.conv2 = TopoCellRTCWNBranchBlock(emb_dim, heads=heads, max_dim=max_dim)
        self.conv3 = TopoCellRTCWNBranchBlock(emb_dim, heads=heads, max_dim=max_dim)
        self.conv4 = TopoCellRTCWNBranchBlock(emb_dim, heads=heads, max_dim=max_dim)
        self.conv5 = TopoCellRTCWNBranchBlock(emb_dim, heads=heads, max_dim=max_dim)
        self.conv6 = TopoCellRTCWNBranchBlock(emb_dim, heads=heads, max_dim=max_dim)

        # 与师兄 TopoCellRTNet 对齐
        self.graphline = nn.Linear(emb_dim * 7, emb_dim * 2)

        self.trans_graph = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim * 4),
            nn.GELU(),
            nn.Linear(emb_dim * 4, emb_dim * 2),
            nn.GELU(),
        )

        self.trans_add = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim * 4),
            nn.GELU(),
            nn.Linear(emb_dim * 4, emb_dim * 2),
        )

        self.trans_out = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim * 4),
            nn.GELU(),
            nn.Linear(emb_dim * 4, emb_dim * 2),
            nn.GELU(),
        )

        self.global_dim = 24
        self.global_proj = nn.Sequential(
            nn.Linear(self.global_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim * 2),
        )

        self.global_gate = nn.Sequential(
            nn.Linear(emb_dim * 4, emb_dim * 2),
            nn.GELU(),
            nn.Linear(emb_dim * 2, emb_dim * 2),
            nn.Sigmoid(),
        )

        self.out_lin = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim // 2),
            nn.GELU(),
            nn.Linear(emb_dim // 2, emb_dim // 4),
            nn.GELU(),
            nn.Linear(emb_dim // 4, emb_dim // 8),
            nn.GELU(),
            nn.Linear(emb_dim // 8, 1),
        )

    def _edge_nodes_from_boundary(self, data):
        boundary = data.cochains[1].boundary_index

        if boundary is None or boundary.numel() == 0:
            raise RuntimeError("1-cochain boundary_index is empty.")

        node_ids = boundary[0].long()
        edge_ids = boundary[1].long()

        num_edges = int(data.cochains[1].num_cells)

        order = torch.argsort(edge_ids)
        node_sorted = node_ids[order]
        edge_sorted = edge_ids[order]

        counts = torch.bincount(edge_sorted, minlength=num_edges)

        if not torch.all(counts == 2):
            bad = torch.nonzero(counts != 2).view(-1)[:10].detach().cpu().tolist()
            raise RuntimeError(f"Some 1-cells do not have exactly 2 boundary nodes: {bad}")

        return node_sorted.view(num_edges, 2)

    def _make_directed_graph(self, data, edge_h):
        edge_nodes = self._edge_nodes_from_boundary(data)

        src = edge_nodes[:, 0]
        dst = edge_nodes[:, 1]

        edge_index_fwd = torch.stack([src, dst], dim=0)
        edge_index_rev = torch.stack([dst, src], dim=0)

        directed_edge_index = torch.cat([edge_index_fwd, edge_index_rev], dim=1)
        directed_edge_attr = torch.cat([edge_h, edge_h], dim=0)

        return directed_edge_index, directed_edge_attr

    def _init_ring_h(self, data, edge_h):
        if data.dimension < 2 or 2 not in data.cochains:
            return None

        ring_cochain = data.cochains[2]
        num_rings = int(ring_cochain.num_cells)

        if num_rings == 0:
            return torch.zeros(0, self.emb_dim, device=edge_h.device)

        boundary = ring_cochain.boundary_index
        if boundary is None or boundary.numel() == 0:
            return torch.zeros(num_rings, self.emb_dim, device=edge_h.device)

        edge_ids = boundary[0].long()
        ring_ids = boundary[1].long()

        ring_h = scatter(
            edge_h[edge_ids],
            ring_ids,
            dim=0,
            dim_size=num_rings,
            reduce="mean",
        )

        return ring_h

    def forward(self, data, include_partial=False):
        x = data.cochains[0].x
        edge_attr = data.cochains[1].x
        batch = data.cochains[0].batch

        h = F.gelu(self.in_node(x))
        edge_h = F.gelu(self.in_edge(edge_attr))
        ring_h = self._init_ring_h(data, edge_h)

        directed_edge_index, directed_edge_attr = self._make_directed_graph(data, edge_h)

        h1, edge_h, ring_h = self.conv1(data, h, edge_h, directed_edge_index, directed_edge_attr, ring_h)
        directed_edge_index, directed_edge_attr = self._make_directed_graph(data, edge_h)

        h2, edge_h, ring_h = self.conv2(data, h1, edge_h, directed_edge_index, directed_edge_attr, ring_h)
        directed_edge_index, directed_edge_attr = self._make_directed_graph(data, edge_h)

        h3, edge_h, ring_h = self.conv3(data, h2, edge_h, directed_edge_index, directed_edge_attr, ring_h)
        directed_edge_index, directed_edge_attr = self._make_directed_graph(data, edge_h)

        h4, edge_h, ring_h = self.conv4(data, h3, edge_h, directed_edge_index, directed_edge_attr, ring_h)
        directed_edge_index, directed_edge_attr = self._make_directed_graph(data, edge_h)

        h5, edge_h, ring_h = self.conv5(data, h4, edge_h, directed_edge_index, directed_edge_attr, ring_h)
        directed_edge_index, directed_edge_attr = self._make_directed_graph(data, edge_h)

        h6, edge_h, ring_h = self.conv6(data, h5, edge_h, directed_edge_index, directed_edge_attr, ring_h)

        hhh1 = torch.cat([h, h1, h2, h3, h4, h5, h6], dim=-1)
        hhh1 = self.graphline(hhh1)

        concat = F.gelu(hhh1)
        concat = self.trans_graph(concat)

        add_x = global_add_pool(concat, batch)

        score = torch.sigmoid(self.trans_add(add_x))
        result = torch.mul(score, add_x)
        result = self.layerNorm_out(result)

        result = self.trans_out(result)

        if hasattr(data, "global_feat") and data.global_feat is not None:
            g = data.global_feat.float().to(result.device)
            g = g.view(g.size(0), -1)

            if g.size(1) == self.global_dim:
                g = self.global_proj(g)
                gate = self.global_gate(torch.cat([result, g], dim=-1))
                result = result + 0.1 * gate * g

        out = self.out_lin(result).view(-1)

        if include_partial:
            return out, {
                "mol_emb": result,
                "edge_h": edge_h,
                "ring_h": ring_h,
                "score": score,
            }

        return out
