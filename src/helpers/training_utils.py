import torch
from torch import Tensor
from sklearn.model_selection import KFold

def sparse_multi_label_k_fold(labels : Tensor, k : int = 4) -> Tensor:
    """
    labels : {-1,0,1}^NxM tensor where N represents the number of datapoints and M the number of labels ,
    l_ij = 0 represents "datapoint i not assigned to label j", l_ij = 1 represents "datapoint i assigned to label j" 
    while -1 represents the absence of an assignment (for semis-supervised settings).

    k : number of folds.
    """

    split_tensor = torch.empty(labels.size(0), dtype=torch.long, device=labels.device).fill_(-1)

    while labels.size(-1) != 0:

        labels_subset = labels[split_tensor == -1]
        counts = torch.sum(labels_subset * (labels_subset != -1), dim=0)

        min_idx = counts.argmin().item()
        pos_indices = labels_subset[:,min_idx].eq(1).argwhere().flatten()

        if len(pos_indices) > 0:

            for fold_id, (_, val_idx) in enumerate(KFold(n_splits=k, shuffle=True).split(pos_indices)):
                split_tensor[pos_indices[val_idx]] = fold_id

        labels = torch.cat([labels[:,0:min_idx], labels[:,(min_idx+1):]], dim=-1)

    ### Handle the case of datapoints assigned to zero labels
    pos_indices = (split_tensor == -1).argwhere().flatten()

    for fold_id, (_, val_idx) in enumerate(KFold(n_splits=k, shuffle=True).split(pos_indices)):
        split_tensor[pos_indices[val_idx]] = fold_id

    return split_tensor