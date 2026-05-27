import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch_geometric.data import Data
import math
from typing import Union, List

import rdkit.Chem as Chem
from rdkit.Chem import Lipinski
from rdkit.Chem import Crippen
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import rdPartialCharges

# 引入项目内部模块
from mp.dataset import InMemoryComplexDataset
from mp.utils import convert_graph_dataset_with_rings
from mp.complex import ComplexBatch

# =========================================================================
# 💡 核心升级区：内嵌 ABCoRT 高精度特征工程 (Feature Engineering)
# =========================================================================

# 🌟 [升级]：加入 is_in_ring_size_n（9D），让每个原子明确感知自己所在的环大小（3~10元环），共 55 维！
atom_features = [
    'chiral_center', 'cip_code', 'crippen_log_p_contrib', 
    'crippen_molar_refractivity_contrib', 'degree', 'element',
    'gasteiger_charge', 'hybridization',
    'is_aromatic', 'is_h_acceptor', 'is_h_donor', 'is_hetero',
    'labute_asa_contrib','num_hs', 'num_valence',
    'tpsa_contrib', 'atom_in_ring', 
    'formal_charge', 'mass',  # <-- 找回丢失的 4 维极其关键的物理量
    'is_in_ring_size_n',  # <-- [新增] 9D 环大小 one-hot：让原子显式感知5元环/6元环/大环等 (46D -> 55D)
]

bond_features = [
    'bondstereo', 'bondtype', 'is_conjugated',
    'is_rotatable', 'bond_dir', 'bond_is_in_ring',
]

def onehot_encode(x: Union[float, int, str], allowable_set: List[Union[float, int, str]]) -> List[float]:
    result = list(map(lambda s: float(x == s), allowable_set))
    return result

def encode(x: Union[float, int, str]) -> List[float]:
    if x is None or np.isnan(x):
        x = 0.0
    return [float(x)]

# ----------------- 原子特征提取函数 -----------------
def is_in_ring(bond: Chem.Bond) -> List[float]:
    return encode(x=bond.IsInRing())

def bondtype(bond: Chem.Bond) -> List[float]:
    return onehot_encode(
        x=bond.GetBondType(),
        allowable_set=[Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                       Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
    )

def is_conjugated(bond):
    return encode(x=bond.GetIsConjugated())

def bond_dir(bond: Chem.Bond) -> List[float]:
    return onehot_encode(
        x=bond.GetBondDir(),
        allowable_set=[Chem.rdchem.BondDir.NONE, Chem.rdchem.BondDir.ENDUPRIGHT, Chem.rdchem.BondDir.ENDDOWNRIGHT]
    )

def is_rotatable(bond: Chem.Bond) -> List[float]:
    mol = bond.GetOwningMol()
    atom_indices = tuple(sorted([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]))
    return encode(x=atom_indices in Lipinski._RotatableBonds(mol))

def bondstereo(bond: Chem.Bond) -> List[float]:
    return onehot_encode(
        x=bond.GetStereo(),
        allowable_set=[Chem.rdchem.BondStereo.STEREONONE, Chem.rdchem.BondStereo.STEREOZ, Chem.rdchem.BondStereo.STEREOE]
    )

def bond_is_in_ring(bond) -> List[float]:
    r_size = 0  
    for ring_size in [10, 9, 8, 7, 6, 5, 4, 3]:
        if bond.IsInRingSize(ring_size): 
            r_size = ring_size
            break
    return onehot_encode(x=r_size, allowable_set=[0, 3, 4, 5, 6, 7, 8, 9, 10])

def ExplicitValence(atom: Chem.Atom) -> List[float]:
    return onehot_encode(x=atom.GetExplicitValence(), allowable_set=[1,2,3,4,5,6])

def ImplicitValence(atom: Chem.Atom) -> List[float]:
    return onehot_encode(x=atom.GetImplicitValence(), allowable_set=[0,1,2,3])

def invert_Chirality(atom: Chem.Atom) -> List[float]:
    return encode(x=atom.InvertChirality())

def Total_degree(atom: Chem.Atom) -> List[float]:
    return onehot_encode(x=atom.GetTotalDegree(), allowable_set=[1, 2, 3, 4])

def Num_ExplicitHs(atom: Chem.Atom) -> List[float]:
    return onehot_encode(x=atom.GetNumExplicitHs(), allowable_set=[0,1])

def atom_in_ring(atom: Chem.Atom) -> List[float]:
    return encode(x=atom.IsInRing())

def chiral_center(atom: Chem.Atom) -> List[float]:
    return encode(x=atom.HasProp("_ChiralityPossible"))

def cip_code(atom: Chem.Atom) -> List[float]:
    if atom.HasProp("_CIPCode"):
        return onehot_encode(x=atom.GetProp("_CIPCode"), allowable_set=["R", "S"])
    return [0.0, 0.0]

def ChiralTag(atom: Chem.Atom) -> List[float]:
    return onehot_encode(
        x=atom.GetChiralTag(),
        allowable_set=[Chem.rdchem.ChiralType.CHI_UNSPECIFIED, Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW]
    )

def element(atom: Chem.Atom) -> List[float]:
    return onehot_encode(
        x=atom.GetSymbol(),
        allowable_set=['H', 'C', 'O', 'S', 'N', 'P', 'F', 'Cl', 'Br', 'I', 'Si']
    )

def hybridization(atom: Chem.Atom) -> List[float]:
    return onehot_encode(
        x=atom.GetHybridization(),
        allowable_set=[Chem.rdchem.HybridizationType.S, Chem.rdchem.HybridizationType.SP,
                       Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3]
    )

def formal_charge(atom: Chem.Atom) -> List[float]:
    return onehot_encode(x=atom.GetFormalCharge(), allowable_set=[-1,0,1])

def mass(atom: Chem.Atom) -> List[float]:
    return encode(x=atom.GetMass() / 100)

def is_aromatic(atom: Chem.Atom) -> List[float]:
    return encode(x=atom.GetIsAromatic())

def num_hs(atom: Chem.Atom) -> List[float]:
    return onehot_encode(x=atom.GetTotalNumHs(), allowable_set=[0, 1, 2, 3])

def num_valence(atom: Chem.Atom) -> List[float]:
    return onehot_encode(x=atom.GetTotalValence(), allowable_set=[1, 2, 3, 4, 5, 6])

def degree(atom: Chem.Atom) -> List[float]:
    return onehot_encode(x=atom.GetDegree(), allowable_set=[1, 2, 3, 4])

def is_in_ring_size_n(atom: Chem.Atom) -> List[float]:
    r_size = 0
    for ring_size in [10, 9, 8, 7, 6, 5, 4, 3]:
        if atom.IsInRingSize(ring_size): 
            r_size = ring_size
            break
    return onehot_encode(x=r_size, allowable_set=[0, 3, 4, 5, 6, 7, 8, 9, 10])

def is_hetero(atom: Chem.Atom) -> List[float]:
    mol = atom.GetOwningMol()
    return encode(x=atom.GetIdx() in [i[0] for i in Lipinski._Heteroatoms(mol)])

def is_h_donor(atom: Chem.Atom) -> List[float]:
    mol = atom.GetOwningMol()
    return encode(x=atom.GetIdx() in [i[0] for i in Lipinski._HDonors(mol)])

def is_h_acceptor(atom: Chem.Atom) -> List[float]:
    mol = atom.GetOwningMol()
    return encode(x=atom.GetIdx() in [i[0] for i in Lipinski._HAcceptors(mol)])

def crippen_log_p_contrib(atom: Chem.Atom) -> List[float]:
    mol = atom.GetOwningMol()
    val = Crippen._GetAtomContribs(mol)[atom.GetIdx()][0]
    return encode(x=val / 10.0)

def crippen_molar_refractivity_contrib(atom: Chem.Atom) -> List[float]:
    mol = atom.GetOwningMol()
    return encode(x=Crippen._GetAtomContribs(mol)[atom.GetIdx()][1])

def tpsa_contrib(atom: Chem.Atom) -> List[float]:
    mol = atom.GetOwningMol()
    val = rdMolDescriptors._CalcTPSAContribs(mol)[atom.GetIdx()]
    return encode(x=val / 100.0)

def labute_asa_contrib(atom: Chem.Atom) -> List[float]:
    mol = atom.GetOwningMol()
    return encode(x=rdMolDescriptors._CalcLabuteASAContribs(mol)[0][atom.GetIdx()])

def gasteiger_charge(atom: Chem.Atom) -> List[float]:
    return encode(x=atom.GetDoubleProp('_GasteigerCharge') if atom.HasProp('_GasteigerCharge') else 0.0)

# ----------------- 总调度提取器 -----------------
def bond_featurizer(bond: Chem.Bond) -> np.ndarray:
    return np.concatenate([globals()[bond_feature](bond) for bond_feature in bond_features], axis=0)

def atom_featurizer(atom: Chem.Atom) -> np.ndarray:
    return np.concatenate([globals()[atom_feature](atom) for atom_feature in atom_features], axis=0)


# =========================================================================
# SMRT 数据集定义
# =========================================================================

class SMRTComplexDataset(InMemoryComplexDataset):
    """
    专门用于 SMRT 数据集的 Dataset 类，无缝内嵌 ABCoRT 高精度提取特征。
    """
    def __init__(self, root, csv_path, max_ring_size=6, use_edge_features=True, 
                 n_jobs=4, init_method='sum', include_down_adj=True):
        self.csv_path = csv_path
        self._max_ring_size = max_ring_size
        self._use_edge_features = use_edge_features
        self._n_jobs = n_jobs
        # 🌟🌟🌟 补上下面这两行 🌟🌟🌟
        self._init_method = init_method
        self.include_down_adj = include_down_adj
        super(SMRTComplexDataset, self).__init__(
            root=root, max_dim=2, init_method=init_method, 
            include_down_adj=include_down_adj, cellular=True
        )
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [os.path.basename(self.csv_path)]

    @property
    def processed_file_names(self):
        return ['smrt_complex.pt']

    @property
    def processed_dir(self):
        # 强制更新缓存文件夹名称，确保提取到真正的 46维/21维 特征
        suffix = f"_r{self._max_ring_size}_Full46D_Embedded"
        if self._use_edge_features: suffix += "_E"
        return os.path.join(self.root, 'processed' + suffix)

    def download(self):
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV file not found at {self.csv_path}")

    def process(self):
        print(f"Processing raw data from: {self.csv_path}")
        try:
            # 1. 尝试标准读取
            df = pd.read_csv(self.csv_path, engine="python")
            
            # 2. 清理列名（去除空格并转小写）
            df.columns = [str(c).lower().strip() for c in df.columns]
            
            # 🌟 3. 核心修复：ABCoRT 官方表的列名是 'smile'，自动把它改为 'smiles'
            if 'smile' in df.columns and 'smiles' not in df.columns:
                df.rename(columns={'smile': 'smiles'}, inplace=True)
                
            # 4. 如果连 smile 也没有，才尝试备选方案
            if 'smiles' not in df.columns or 'rt' not in df.columns:
                df = pd.read_csv(self.csv_path, sep=r"\s+", names=["smiles", "rt"], header=0, engine="python")
                
            # 5. 过滤死体积
            df = df[df['rt'] > 300.0]
            print(f"✅ 成功读取数据集，有效分子数: {len(df)}")
            
        except Exception as e:
            print(f"❌ 读取 CSV 彻底失败: {e}")
            raise e  # 🚨 必须使用 raise e 强制让程序在这里崩溃，绝对不能用 return！
            
        data_list = []
        print("Step 1/2: Extracting EXACT ABCoRT Features (Full 46D Atoms, 21D Bonds)...")
        
        for index, row in tqdm(df.iterrows(), total=len(df)):
            smiles = row.get("smiles", None)
            rt = row.get("rt", None)
            
            if pd.isna(smiles): continue
            smiles = str(smiles)
            try:
                rt = float(rt)
            except (ValueError, TypeError):
                continue
                
            mol = Chem.MolFromSmiles(smiles)
            if mol is None: continue
            
            # 🌟🌟🌟 [核心修复] 必须强制计算手性和电荷，否则提取出来全是 0！ 🌟🌟🌟
            Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
            try:
                rdPartialCharges.ComputeGasteigerCharges(mol)
            except:
                pass # 忽略少数无法计算电荷的奇葩分子

            # --- A. 构建原汁原味的 46 维原子特征 ---
            atom_features_list = []
            for atom in mol.GetAtoms():
                atom_feat = atom_featurizer(atom)
                atom_features_list.append(atom_feat)
                
            x = torch.tensor(np.array(atom_features_list), dtype=torch.float)

            # --- B. 构建原汁原味的 21 维边特征 ---
            edge_index_list = []
            edge_attr_list = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                
                edge_feature = bond_featurizer(bond)

                edge_index_list.append([i, j])
                edge_attr_list.append(edge_feature)
                edge_index_list.append([j, i])
                edge_attr_list.append(edge_feature)

            if len(edge_index_list) == 0:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_attr = torch.empty((0, 21), dtype=torch.float)
            else:
                edge_index = torch.tensor(edge_index_list).t().contiguous()
                edge_attr = torch.tensor(np.array(edge_attr_list), dtype=torch.float) 

            # --- C. 构建 Data 对象 ---
            y = torch.tensor([rt], dtype=torch.float)
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, 
                        y=y, num_nodes=len(x))
            data_list.append(data)

        print(f"Step 2/2: Lifting Graphs to Cell Complexes (finding rings <= {self._max_ring_size})...")
        complexes, _, _ = convert_graph_dataset_with_rings(
            data_list,
            max_ring_size=self._max_ring_size,
            include_down_adj=self.include_down_adj,
            init_method=self._init_method,
            init_edges=self._use_edge_features,
            init_rings=False, 
            n_jobs=self._n_jobs
        )

        print(f"Saving processed data to {self.processed_paths[0]}...")
        torch.save(self.collate(complexes, self.max_dim), self.processed_paths[0])