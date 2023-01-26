from typing import List, Optional, Tuple, NamedTuple
from torch import Tensor

class Adj(NamedTuple):
    edge_index: Tensor
    edge_type: Optional[Tensor]
    edge_weight: Optional[Tensor]       # Time offset relevant to 
    orig_seq: Optional[Tensor]
    # time_diff: Optional[Tensor]
    size: Optional[Tuple[int, int]]

    def to(self, *args, **kwargs):
        edge_index = self.edge_index.to(*args, **kwargs)
        edge_type = self.edge_type.to(*args, **kwargs) if self.edge_type is not None else None
        edge_weight = self.edge_weight.to(*args, **kwargs) if self.edge_weight is not None else None
        orig_seq = self.orig_seq.to(*args, **kwargs) if self.orig_seq is not None else None
        # time_diff = self.time_diff.to(*args, **kwargs) if self.time_diff is not None else None
        # return Adj(edge_index, edge_type, edge_weight, orig_seq, time_diff, self.size)
        return Adj(edge_index, edge_type, edge_weight, orig_seq, self.size)

class Adj_v2(NamedTuple):
    edge_index: Tensor
    edge_type: Optional[Tensor]
    edge_weight: Optional[List[Tensor]]       # Time offset relevant to 
    orig_seq: Optional[List[Tensor]]
    # time_diff: Optional[Tensor]
    size: Optional[Tuple[int, int]]

    def to(self, *args, **kwargs):
        edge_index = self.edge_index.to(*args, **kwargs)
        edge_type = self.edge_type.to(*args, **kwargs) if self.edge_type is not None else None
        edge_weight = [self.edge_weight[i].to(*args, **kwargs) for i in range(len(self.edge_weight))] if self.edge_weight is not None else None
        orig_seq =  [self.orig_seq[i].to(*args, **kwargs) for i in range(len(self.orig_seq))] if self.orig_seq is not None else None
        # time_diff = self.time_diff.to(*args, **kwargs) if self.time_diff is not None else None
        # return Adj(edge_index, edge_type, edge_weight, orig_seq, time_diff, self.size)
        return Adj_v2(edge_index, edge_type, edge_weight, orig_seq, self.size)


class STP_Adj(NamedTuple):
    edge_index: Tensor
    node_idx: Tensor
    center_node: Tensor

    def to(self, *args, **kwargs):
        edge_index = self.edge_index.to(*args, **kwargs)
        node_idx = self.node_idx.to(*args, **kwargs)
        center_node = self.center_node.to(*args, **kwargs)
        return STP_Adj(edge_index, node_idx, center_node)