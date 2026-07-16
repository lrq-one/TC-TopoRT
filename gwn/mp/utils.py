import torch
import gudhi as gd
import itertools
import networkx as nx
import numpy as np

from tqdm import tqdm
from mp.complex import Cochain, Complex
from typing import List, Dict, Optional, Union
from torch import Tensor
from torch_geometric.typing import Adj
from torch_scatter import scatter
from mp.parallel import ProgressParallel
from joblib import delayed


def pyg_to_simplex_tree(edge_index: Tensor, size: int):
    """Constructs a simplex tree from a PyG graph."""
    st = gd.SimplexTree()
    for v in range(size):
        st.insert([v])

    edges = edge_index.numpy()
    for e in range(edges.shape[1]):
        edge = [edges[0][e], edges[1][e]]
        st.insert(edge)

    return st


def get_simplex_boundaries(simplex):
    boundaries = itertools.combinations(simplex, len(simplex) - 1)
    return [tuple(boundary) for boundary in boundaries]


def build_tables(simplex_tree, size):
    complex_dim = simplex_tree.dimension()
    id_maps = [{} for _ in range(complex_dim+1)]
    simplex_tables = [[] for _ in range(complex_dim+1)]
    boundaries_tables = [[] for _ in range(complex_dim+1)]

    simplex_tables[0] = [[v] for v in range(size)]
    id_maps[0] = {tuple([v]): v for v in range(size)}

    for simplex, _ in simplex_tree.get_simplices():
        dim = len(simplex) - 1
        if dim == 0:
            continue
        next_id = len(simplex_tables[dim])
        id_maps[dim][tuple(simplex)] = next_id
        simplex_tables[dim].append(simplex)

    return simplex_tables, id_maps


def extract_boundaries_and_coboundaries_from_simplex_tree(simplex_tree, id_maps, complex_dim: int):
    boundaries = [{} for _ in range(complex_dim+2)]
    coboundaries = [{} for _ in range(complex_dim+2)]
    boundaries_tables = [[] for _ in range(complex_dim+1)]

    for simplex, _ in simplex_tree.get_simplices():
        simplex_dim = len(simplex) - 1
        level_coboundaries = coboundaries[simplex_dim]
        level_boundaries = boundaries[simplex_dim + 1]

        if simplex_dim > 0:
            boundaries_ids = [id_maps[simplex_dim-1][boundary] for boundary in get_simplex_boundaries(simplex)]
            boundaries_tables[simplex_dim].append(boundaries_ids)

        simplex_coboundaries = simplex_tree.get_cofaces(simplex, codimension=1)
        for coboundary, _ in simplex_coboundaries:
            if tuple(simplex) not in level_coboundaries:
                level_coboundaries[tuple(simplex)] = list()
            level_coboundaries[tuple(simplex)].append(tuple(coboundary))

            if tuple(coboundary) not in level_boundaries:
                level_boundaries[tuple(coboundary)] = list()
            level_boundaries[tuple(coboundary)].append(tuple(simplex))

    return boundaries_tables, boundaries, coboundaries


def build_adj(boundaries: List[Dict], coboundaries: List[Dict], id_maps: List[Dict], complex_dim: int,
              include_down_adj: bool):
    def initialise_structure():
        return [[] for _ in range(complex_dim+1)]

    upper_indexes, lower_indexes = initialise_structure(), initialise_structure()
    all_shared_boundaries, all_shared_coboundaries = initialise_structure(), initialise_structure()

    for dim in range(complex_dim+1):
        for simplex, id in id_maps[dim].items():
            if dim > 0:
                for boundary1, boundary2 in itertools.combinations(boundaries[dim][simplex], 2):
                    id1, id2 = id_maps[dim - 1][boundary1], id_maps[dim - 1][boundary2]
                    upper_indexes[dim - 1].extend([[id1, id2], [id2, id1]])
                    all_shared_coboundaries[dim - 1].extend([id, id])

            if include_down_adj and dim < complex_dim and simplex in coboundaries[dim]:
                for coboundary1, coboundary2 in itertools.combinations(coboundaries[dim][simplex], 2):
                    id1, id2 = id_maps[dim + 1][coboundary1], id_maps[dim + 1][coboundary2]
                    lower_indexes[dim + 1].extend([[id1, id2], [id2, id1]])
                    all_shared_boundaries[dim + 1].extend([id, id])

    return all_shared_boundaries, all_shared_coboundaries, lower_indexes, upper_indexes


def construct_features(vx: Tensor, cell_tables, init_method: str) -> List:
    features = [vx]
    for dim in range(1, len(cell_tables)):
        aux_1 = []
        aux_0 = []
        for c, cell in enumerate(cell_tables[dim]):
            aux_1 += [c for _ in range(len(cell))]
            aux_0 += cell
        node_cell_index = torch.LongTensor([aux_0, aux_1])
        in_features = vx.index_select(0, node_cell_index[0])
        features.append(scatter(in_features, node_cell_index[1], dim=0,
                                dim_size=len(cell_tables[dim]), reduce=init_method))
    return features


def extract_labels(y, size):
    v_y, complex_y = None, None
    if y is None:
        return v_y, complex_y
    y_shape = list(y.size())
    if y_shape[0] == 1:
        complex_y = y
    else:
        v_y = y
    return v_y, complex_y


def generate_cochain(dim, x, all_upper_index, all_lower_index,
                   all_shared_boundaries, all_shared_coboundaries, cell_tables, boundaries_tables,
                   complex_dim, y=None):
    num_cells_down = len(cell_tables[dim-1]) if dim > 0 else None
    num_cells_up = len(cell_tables[dim+1]) if dim < complex_dim else 0

    up_index = (torch.tensor(all_upper_index[dim], dtype=torch.long).t()
                if len(all_upper_index[dim]) > 0 else None)
    down_index = (torch.tensor(all_lower_index[dim], dtype=torch.long).t()
                  if len(all_lower_index[dim]) > 0 else None)
    shared_coboundaries = (torch.tensor(all_shared_coboundaries[dim], dtype=torch.long)
                      if len(all_shared_coboundaries[dim]) > 0 else None)
    shared_boundaries = (torch.tensor(all_shared_boundaries[dim], dtype=torch.long)
                    if len(all_shared_boundaries[dim]) > 0 else None)
    
    boundary_index = None
    if len(boundaries_tables[dim]) > 0:
        boundary_index = [list(), list()]
        for s, cell in enumerate(boundaries_tables[dim]):
            for boundary in cell:
                boundary_index[1].append(s)
                boundary_index[0].append(boundary)
        boundary_index = torch.LongTensor(boundary_index)

    return Cochain(dim=dim, x=x, upper_index=up_index,
                 lower_index=down_index, shared_coboundaries=shared_coboundaries,
                 shared_boundaries=shared_boundaries, y=y, num_cells_down=num_cells_down,
                 num_cells_up=num_cells_up, boundary_index=boundary_index)


# ==================================================================================
#  Modified Logic: Use NetworkX instead of graph-tool to avoid SegFaults
# ==================================================================================

def get_rings(edge_index, max_k=7):

    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.numpy()

    rows, cols = edge_index
    g = nx.Graph()
    edges = list(zip(rows, cols))
    g.add_edges_from(edges)
    
    cycles = nx.minimum_cycle_basis(g)
    
    rings = []
    for cycle in cycles:

        if 3 <= len(cycle) <= max_k:
            rings.append(tuple(sorted(cycle)))

    return list(set(rings))


def build_tables_with_rings(edge_index, simplex_tree, size, max_k):

    cell_tables, id_maps = build_tables(simplex_tree, size)
    

    rings = get_rings(edge_index, max_k=max_k)
    
    if len(rings) > 0:
        id_maps += [{}]
        cell_tables += [[]]
        for cell in rings:
            next_id = len(cell_tables[2])
            id_maps[2][cell] = next_id
            cell_tables[2].append(list(cell))

    return cell_tables, id_maps


def get_ring_boundaries(ring):
    boundaries = list()
    for n in range(len(ring)):
        a = n
        b = 0 if n + 1 == len(ring) else n + 1
        boundaries.append(tuple(sorted([ring[a], ring[b]])))
    return sorted(boundaries)


def extract_boundaries_and_coboundaries_with_rings(simplex_tree, id_maps):
    boundaries_tables, boundaries, coboundaries = extract_boundaries_and_coboundaries_from_simplex_tree(
                                                    simplex_tree, id_maps, simplex_tree.dimension())
    
    if len(id_maps) == 3:
        boundaries += [{}]
        coboundaries += [{}]
        boundaries_tables += [[]]
        
        for cell in id_maps[2]: # cell 是环的元组 (node_idx1, node_idx2, ...)
            cell_boundaries = get_ring_boundaries(cell)
            boundaries[2][cell] = list()
            boundaries_tables[2].append([])
            
            for boundary in cell_boundaries:
                if boundary not in id_maps[1]:

                    continue
                boundary_id = id_maps[1][boundary]
                boundaries[2][cell].append(boundary)
                boundaries_tables[2][-1].append(boundary_id)
                
                if boundary not in coboundaries[1]:
                    coboundaries[1][boundary] = list()
                coboundaries[1][boundary].append(cell)
    
    return boundaries_tables, boundaries, coboundaries


def compute_ring_2complex(x: Union[Tensor, np.ndarray], edge_index: Union[Tensor, np.ndarray],
                          edge_attr: Optional[Union[Tensor, np.ndarray]],
                          size: int, y: Optional[Union[Tensor, np.ndarray]] = None, 
                          global_feat: Optional[Tensor] = None,
                          max_k: int = 7,
                          include_down_adj=True, init_method: str = 'sum',
                          init_edges=True, init_rings=False) -> Complex:
    
    if isinstance(x, np.ndarray): x = torch.tensor(x)
    if isinstance(edge_index, np.ndarray): edge_index = torch.tensor(edge_index)
    if isinstance(edge_attr, np.ndarray): edge_attr = torch.tensor(edge_attr)
    if isinstance(y, np.ndarray): y = torch.tensor(y)
    if isinstance(global_feat, np.ndarray): global_feat = torch.tensor(global_feat)

    simplex_tree = pyg_to_simplex_tree(edge_index, size)

    cell_tables, id_maps = build_tables_with_rings(edge_index, simplex_tree, size, max_k)
    complex_dim = len(id_maps)-1

    boundaries_tables, boundaries, co_boundaries = extract_boundaries_and_coboundaries_with_rings(simplex_tree, id_maps)


    shared_boundaries, shared_coboundaries, lower_idx, upper_idx = build_adj(
        boundaries, co_boundaries, id_maps, complex_dim, include_down_adj)
    
    xs = [x, None, None]
    constructed_features = construct_features(x, cell_tables, init_method)
    
    if init_rings and len(constructed_features) > 2:
        xs[2] = constructed_features[2]
    elif len(cell_tables) > 2:
        xs[2] = torch.zeros(len(cell_tables[2]), x.size(1))
    
    if init_edges and complex_dim >= 1:
        if edge_attr is None:
            xs[1] = constructed_features[1]
        else:
            if edge_attr.dim() == 1: edge_attr = edge_attr.view(-1, 1)
            num_edges = len(id_maps[1])
            edge_feat_dim = edge_attr.size(1)
            xs[1] = torch.zeros(num_edges, edge_feat_dim)
            
            edge_feat_map = {}
            row, col = edge_index
            for k in range(row.size(0)):
                u, v = row[k].item(), col[k].item()
                canon = tuple(sorted((u, v)))
                edge_feat_map[canon] = edge_attr[k]
                
            for edge_tuple, edge_id in id_maps[1].items():
                if edge_tuple in edge_feat_map:
                    xs[1][edge_id] = edge_feat_map[edge_tuple]
                    
    v_y, complex_y = extract_labels(y, size)
    cochains = []
    for i in range(complex_dim + 1):
        target = v_y if i == 0 else None
        if xs[i] is None: 
            num_entities = len(cell_tables[i])
            if i == 1 and edge_attr is not None:
                feat_dim = edge_attr.size(1) if edge_attr.dim() > 1 else 1
            else:
                feat_dim = x.size(1)
            xs[i] = torch.zeros(num_entities, feat_dim, dtype=torch.float)
            
        cochain = generate_cochain(i, xs[i], upper_idx, lower_idx, shared_boundaries, shared_coboundaries,
                               cell_tables, boundaries_tables, complex_dim=complex_dim, y=target)
        cochains.append(cochain)

    return Complex(*cochains, y=complex_y, dimension=complex_dim, global_feat=global_feat)


def convert_graph_dataset_with_rings(dataset, max_ring_size=7, include_down_adj=False,
                                     init_method: str = 'sum', init_edges=True, init_rings=False,
                                     n_jobs=1):
    

    def process_one(data):
        global_feat = getattr(data, 'global_feat', None)
        return compute_ring_2complex(
            data.x, data.edge_index, data.edge_attr,
            data.num_nodes, y=data.y, global_feat=global_feat,
            max_k=max_ring_size,
            include_down_adj=include_down_adj, init_method=init_method,
            init_edges=init_edges, init_rings=init_rings)

    if n_jobs > 1:
        parallel = ProgressParallel(n_jobs=n_jobs, use_tqdm=True, total=len(dataset))
        complexes = parallel(delayed(process_one)(data) for data in dataset)
    else:
        complexes = [process_one(data) for data in tqdm(dataset)]

    return complexes, -1, None
