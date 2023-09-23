import os.path as osp

from torch_geometric.datasets import Planetoid, CitationFull, WikiCS, Coauthor, Amazon, AttributedGraphDataset
import torch_geometric.transforms as T

from ogb.nodeproppred import PygNodePropPredDataset

def get_dataset(path, name):
    assert name in ['Cora', 'CiteSeer', 'PubMed', 'DBLP', 'Karate', 'WikiCS', 'Coauthor-CS', 'Coauthor-Phy',
                    'Computers', 'Photo', 'ogbn-arxiv', 'ogbg-code', 'BlogCatalog', 'Flickr', 'sbm']
    name = 'dblp' if name == 'DBLP' else name
    root_path = osp.expanduser('~/datasets')

    if name == 'Coauthor-CS':
        return Coauthor(root=path, name='cs', transform=T.NormalizeFeatures())

    if name == 'Coauthor-Phy':
        return Coauthor(root=path, name='physics', transform=T.NormalizeFeatures())

    if name == 'WikiCS':
        return WikiCS(root=osp.join(root_path, name))

    if name == 'Computers':
        return Amazon(root=root_path, name='computers', transform=T.NormalizeFeatures())

    if name == 'Photo':
        return Amazon(root=root_path, name='photo', transform=T.NormalizeFeatures())
    if name in ['BlogCatalog']:
        return AttributedGraphDataset(root=root_path, name=name, transform=T.NormalizeFeatures())
    if name in ['Flickr']:
        return AttributedGraphDataset(root=root_path, name=name, transform=NormalizeFeaturesSparse())
    if name.startswith('ogbn'):
        return PygNodePropPredDataset(root=osp.join(root_path, 'OGB'), name=name, transform=T.NormalizeFeatures())

    return (CitationFull if name == 'dblp' else Planetoid)(osp.join(root_path, 'Citation'), name, transform=T.NormalizeFeatures()) # public split

from typing import List, Union

from torch_geometric.data import Data, HeteroData
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform


@functional_transform('normalize_features_sparse')
class NormalizeFeaturesSparse(BaseTransform):
    r"""Row-normalizes the attributes given in :obj:`attrs` to sum-up to one
    (functional name: :obj:`normalize_features`).

    Args:
        attrs (List[str]): The names of attributes to normalize.
            (default: :obj:`["x"]`)
    """
    def __init__(self, attrs: List[str] = ["x"]):
        self.attrs = attrs

    def __call__(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        for store in data.stores:
            for key, value in store.items(*self.attrs):
                value = value.to_dense()
                value = value - value.min()
                value.div_(value.sum(dim=-1, keepdim=True).clamp_(min=1.))
                store[key] = value
        return data


def get_path(base_path, name):
    if name in ['Cora', 'CiteSeer', 'PubMed']:
        return base_path
    else:
        return osp.join(base_path, name)
