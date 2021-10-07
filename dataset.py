import os
import os.path as osp
import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
import networkx as nx
import random
import numpy as np

import pandas as pd
from tqdm import tqdm
from utils import Complete, angle, area_triangle, cal_dist

try:
    from rdkit import Chem
    from rdkit.Chem import ChemicalFeatures
    from rdkit import RDConfig

    fdef_name = osp.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)
except:
    Chem, ChemicalFeatures, RDConfig, fdef_name, chem_feature_factory = 5 * [None]
    print('Please install rdkit for data processing')


class SpecifyTarget(object):
    def __init__(self, target):
        self.target = target

    def __call__(self, data):
        data.y = data.y[self.target].view(-1)
        return data

def load_dataset(path, transform=None):
    dataset = QM9Dataset(path, transform=transform)
    # shuffle data
    indices = list(range(len(dataset)))
    random.seed(1234)
    random.shuffle(indices)
    dataset = dataset[indices]

    # split data
    one_tenth = len(dataset) // 10
    test_dataset = dataset[: one_tenth]
    valid_dataset = dataset[one_tenth: one_tenth * 2]
    train_dataset = dataset[one_tenth * 2:]

    assert len(train_dataset) + len(valid_dataset) + len(test_dataset) == len(dataset)

    return train_dataset, valid_dataset, test_dataset

class QM9Dataset(InMemoryDataset):

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        # 进去基类Dataset里面，首先判断self.processed_paths是否存在
        # 如果不存在就调用process函数，如果存在了不调用process
        super(QM9Dataset, self).__init__(root, transform, pre_transform, pre_filter)
        # 导入数据
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # 通过self.raw_paths获取
        return ['gdb9.sdf', 'gdb9.sdf.csv']

    @property
    def processed_file_names(self):
        # 通过self.processed_paths获取
        return 'processed_qm9.pt'
        

    def download(self):
        pass

    def process(self):
        # data_path: gdb9.sdf
        # target_path: gdb9.sdf.csv
        data_path = self.raw_paths[0]
        target_path = self.raw_paths[1]
        self.property_names = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'u0',
                               'u298', 'h298', 'g298', 'cv']
        self.target = pd.read_csv(target_path, index_col='mol_id')
        # 只取出self.property_names中的对应的列
        self.target = self.target[self.property_names]
        supplier = Chem.SDMolSupplier(data_path, removeHs=False)
        data_list = []
        for i, mol in tqdm(enumerate(supplier)):
            data = self.mol2graph(mol)
            if data is not None:
                data.y = torch.FloatTensor(self.target.iloc[i, :])
                data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # 把所有的数据拼接成1个大数据，另外返回slices用于还原出单个数据
        data, slices = self.collate(data_list)

        print(data)
        print(slices)

        torch.save((data, slices), self.processed_paths[0])

    def get_nodes(self, g):
        feat = []
        for n, d in g.nodes(data=True):
            h_t = []
            h_t += [int(d['a_type'] == x) for x in ['H', 'C', 'N', 'O', 'F', 'Cl', 'S']]
            h_t.append(d['a_num'])
            h_t.append(d['acceptor'])
            h_t.append(d['donor'])
            h_t.append(int(d['aromatic']))
            h_t += [int(d['hybridization'] == x) \
                    for x in (Chem.rdchem.HybridizationType.SP, \
                              Chem.rdchem.HybridizationType.SP2,
                              Chem.rdchem.HybridizationType.SP3)]
            h_t.append(d['num_h'])
            # 5 more
            h_t.append(d['formal_charge'])
            h_t.append(d['explicit_valence'])
            h_t.append(d['implicit_valence'])
            h_t.append(d['num_explicit_hs'])
            h_t.append(d['num_radical_electrons'])
            feat.append((n, h_t))
        feat.sort(key=lambda item: item[0])
        node_attr = torch.FloatTensor([item[1] for item in feat])
        return node_attr

    def get_edges(self, g):
        e = {}
        for n1, n2, d in g.edges(data=True):
            e_t = [int(d['b_type'] == x)
                   for x in (Chem.rdchem.BondType.SINGLE, \
                             Chem.rdchem.BondType.DOUBLE, \
                             Chem.rdchem.BondType.TRIPLE, \
                             Chem.rdchem.BondType.AROMATIC)]
            e_t.append(d['anglemax'])
            e_t.append(d['anglesum'])
            e_t.append(d['anglemean'])

            e_t.append(d['areamax'])
            e_t.append(d['areasum'])
            e_t.append(d['areamean'])

            e_t.append(d['dikmax'])
            e_t.append(d['diksum'])
            e_t.append(d['dikmean'])
            e_t.append(d['dij1'])
            e_t.append(d['dij2'])

            e[(n1, n2)] = e_t
        edge_index = torch.LongTensor(list(e.keys())).transpose(0, 1)
        edge_attr = torch.FloatTensor(list(e.values()))
        return edge_index, edge_attr

    def mol2graph(self, mol):
        if mol is None: return None

        g = nx.DiGraph()

        # Create nodes
        assert len(mol.GetConformers()) == 1
        geom = mol.GetConformers()[0].GetPositions()
        for i in range(mol.GetNumAtoms()):
            atom_i = mol.GetAtomWithIdx(i)
            g.add_node(i,
                       a_type=atom_i.GetSymbol(),
                       a_num=atom_i.GetAtomicNum(),
                       acceptor=0,  # 0 for placeholder
                       donor=0,
                       aromatic=atom_i.GetIsAromatic(),
                       hybridization=atom_i.GetHybridization(),
                       num_h=atom_i.GetTotalNumHs(includeNeighbors=True),
                       # 5 more node features
                       formal_charge=atom_i.GetFormalCharge(),
                       explicit_valence=atom_i.GetExplicitValence(),
                       implicit_valence=atom_i.GetImplicitValence(),
                       num_explicit_hs=atom_i.GetNumExplicitHs(),
                       num_radical_electrons=atom_i.GetNumRadicalElectrons(),
                       )
        # 使用g.nodes.data可以到节点编号以及节点对应的特征

        # Electron donor and acceptor
        # 前面已经初始化了acceptor和donor属性
        # 这里是要给acceptor和donor属性赋值
        fdef_name = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
        factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)
        feats = factory.GetFeaturesForMol(mol)
        for f in range(len(feats)):
            if feats[f].GetFamily() == 'Donor':
                for atom_id in feats[f].GetAtomIds():
                    g.nodes[atom_id]['donor'] = 1
            elif feats[f].GetFamily() == 'Acceptor':
                for atom_id in feats[f].GetAtomIds():
                    g.nodes[atom_id]['acceptor'] = 1

        # Read Edges
        for i in range(mol.GetNumAtoms()):
            for j in range(mol.GetNumAtoms()):
                # i原子和j原子是否相连
                e_ij = mol.GetBondBetweenAtoms(i, j)
                if e_ij is not None:
                    # cal angle and area
                    assert mol.GetNumAtoms() == len(geom)
                    # 这里的angles，areas，dists均是列表，原因是
                    # 后面我们是通过对列表的数据进行统计来得到边的
                    # 属性的
                    angles_ijk = []
                    areas_ijk = []
                    dists_ik = []
                    # 如果i原子和j原子相连，那么获取j原子的所有邻居
                    for neighbor in mol.GetAtomWithIdx(j).GetNeighbors():
                        # 获取j原子的邻居k的索引
                        k = neighbor.GetIdx()
                        # k原子和i原子不能是同一个原子
                        if mol.GetBondBetweenAtoms(j, k) is not None and i != k:
                            # geom: (num_atoms, 3), 3表示3维空间
                            # vector1: i指向j
                            # vector2：i指向k
                            vector1 = geom[j] - geom[i]
                            vector2 = geom[k] - geom[i]
                            # 计算两个向量夹角
                            angles_ijk.append(angle(vector1, vector2))
                            # 计算两个向量围城三角形的面积
                            areas_ijk.append(area_triangle(vector1, vector2))
                            # 原子i和原子k的距离
                            dists_ik.append(cal_dist(geom[i], geom[k]))
                    angles_ijk = np.array(angles_ijk) if angles_ijk != [] else np.array([0.])
                    areas_ijk = np.array(areas_ijk) if areas_ijk != [] else np.array([0.])
                    dists_ik = np.array(dists_ik) if dists_ik != [] else np.array([0.])
                    dist_ij1 = cal_dist(geom[i], geom[j], ord=1)
                    dist_ij2 = cal_dist(geom[i], geom[j], ord=2)

                    g.add_edge(i, j,
                               b_type=e_ij.GetBondType(),

                               anglemax=angles_ijk.max(),
                               anglesum=angles_ijk.sum(),
                               anglemean=angles_ijk.mean(),

                               areamax=areas_ijk.max(),
                               areasum=areas_ijk.sum(),
                               areamean=areas_ijk.mean(),

                               dikmax=dists_ik.max(),
                               diksum=dists_ik.sum(),
                               dikmean=dists_ik.mean(),
                               dij1=dist_ij1,
                               dij2=dist_ij2,
                               )

        # Build pyg data
        node_attr = self.get_nodes(g)
        edge_index, edge_attr = self.get_edges(g)
        pos = torch.FloatTensor(geom)
        data = Data(
            x=node_attr,
            pos=pos,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=None,  # None as a placeholder
            # name=mol.GetProp('_Name'),
        )
        return data


if __name__ == '__main__':
    dataset = QM9Dataset('./data/qm9')
