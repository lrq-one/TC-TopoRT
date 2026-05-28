from typing import  Union, List, Tuple
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
import os
import pickle
from topocell_features import build_atom_bond_graph_features


def compute_topocell_context(smiles):
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return torch.zeros(24, dtype=torch.float32), torch.tensor([0.0], dtype=torch.float32)

    atoms = list(mol.GetAtoms())

    hetero_count = sum(1 for a in atoms if a.GetAtomicNum() not in [1, 6])
    heavy_count = mol.GetNumHeavyAtoms()
    halogen_count = sum(1 for a in atoms if a.GetSymbol() in ["F", "Cl", "Br", "I"])

    aromatic_ring_count = rdMolDescriptors.CalcNumAromaticRings(mol)
    aliphatic_ring_count = rdMolDescriptors.CalcNumAliphaticRings(mol)
    ring_count = rdMolDescriptors.CalcNumRings(mol)

    smarts = {
        "cf3": "[CX4](F)(F)F",
        "sulfonamide": "S(=O)(=O)N",
        "amide": "C(=O)N",
        "urea": "NC(=O)N",
        "piperazine": "N1CCNCC1",
        "morpholine": "O1CCNCC1",
        "imidic": "N=C(O)",
    }

    flags = []
    for patt in smarts.values():
        q = Chem.MolFromSmarts(patt)
        flags.append(float(mol.HasSubstructMatch(q)) if q is not None else 0.0)

    feat = [
        Descriptors.MolWt(mol) / 600.0,
        Crippen.MolLogP(mol) / 10.0,
        rdMolDescriptors.CalcTPSA(mol) / 200.0,
        Lipinski.NumHAcceptors(mol) / 12.0,
        Lipinski.NumHDonors(mol) / 8.0,
        Lipinski.NumRotatableBonds(mol) / 20.0,
        ring_count / 8.0,
        aromatic_ring_count / 6.0,
        aliphatic_ring_count / 6.0,
        rdMolDescriptors.CalcFractionCSP3(mol),
        heavy_count / 80.0,
        hetero_count / max(heavy_count, 1),
        halogen_count / 8.0,
    ]

    feat.extend(flags)

    while len(feat) < 24:
        feat.append(0.0)

    hard_flag = float(
        aromatic_ring_count >= 2
        or ring_count >= 3
        or hetero_count >= 4
        or halogen_count >= 2
        or any(flags)
    )

    return torch.tensor(feat[:24], dtype=torch.float32), torch.tensor([hard_flag], dtype=torch.float32)


class TopoCellRTTrainDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, 'raw')


    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, 'processed')


    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return 'SMRT_train.csv'

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return ['data.pt']

    def download(self):
        pass


    def process(self):

        train_csv = os.environ.get("TOPOCELLRT_TRAIN_CSV", "./SMRT_data/data/SMRT_train.csv")
        print("loading train csv:", train_csv)
        res = pd.read_csv(train_csv)
        y = res['rt']
        smile_list = res['smile']
        data_list = []
        succ_inchi, succ_rt, success_index, atom_feature,edge_index,edge_attr = build_atom_bond_graph_features(smile_list, y,type=0)

        for k, raw_index in enumerate(success_index):
            smile = smile_list.iloc[raw_index]
            rt = float(res['rt'].iloc[raw_index])

            global_feat, hard_flag = compute_topocell_context(smile)

            data = Data(
                x=atom_feature[k],
                y=torch.tensor(rt, dtype=torch.float32),
                edge_index=edge_index[k],
                edge_attr=edge_attr[k],
                global_feat=global_feat.view(1, -1),
                hard_flag=hard_flag.view(1),
                smiles=smile,
            )
            data_list.append(data)

        print(data_list.__len__())
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_filter is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        torch.save((data, slices), self.processed_paths[0])


class TopoCellRTTestDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])


    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, 'raw')


    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, 'processed')


    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return 'SMRT_test.csv'

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return ['data.pt']

    def download(self):
        pass


    def process(self):
        test_csv = os.environ.get("TOPOCELLRT_TEST_CSV", "./SMRT_data/data/SMRT_test.csv")
        print("loading test csv:", test_csv)
        res = pd.read_csv(test_csv)
        y = res['rt']
        smile_list = res['smile']
        succ_inchi, succ_rt, success_index, atom_feature,edge_index,edge_attr = build_atom_bond_graph_features(smile_list,y,type=0)
        data_list = []
        for k, raw_index in enumerate(success_index):
            smile = smile_list.iloc[raw_index]
            rt = float(res['rt'].iloc[raw_index])

            global_feat, hard_flag = compute_topocell_context(smile)

            data = Data(
                x=atom_feature[k],
                y=torch.tensor(rt, dtype=torch.float32),
                edge_index=edge_index[k],
                edge_attr=edge_attr[k],
                global_feat=global_feat.view(1, -1),
                hard_flag=hard_flag.view(1),
                smiles=smile,
            )
            data_list.append(data)

        print(data_list.__len__())
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_filter is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


if __name__ == '__main__':

    print('Loading ...get SMRT train feature')
    dataset1 = TopoCellRTTrainDataset('./SMRT_data/reload/SMRT_train')
    print('Number of graphs in dataset: ', len(dataset1))
    print('Loading ...get SMRT test feature')
    dataset2 = TopoCellRTTestDataset('./SMRT_data/reload/SMRT_test')
    print('Number of graphs in dataset: ', len(dataset2))
