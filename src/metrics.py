import numpy as np
from collections import defaultdict

class FenwickTree:
    def __init__(self, size):
        self.tree = np.zeros(size + 1, dtype=float)

    def update(self, i, delta):
        while i < len(self.tree):
            self.tree[i] += delta
            i += i & (-i)

    def query(self, i):
        s = 0.0
        while i > 0:
            s += self.tree[i]
            i -= i & (-i)
        return s
    
    def range_query(self, i, j):
        if i > j:
            return 0.0
        return self.query(j) - self.query(i - 1)

def get_auc(y_pred, y_true):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    n = len(y_true)
    if n < 2:
        return np.nan

    sorted_unique_preds = np.unique(y_pred)
    pred_to_rank = {val: i + 1 for i, val in enumerate(sorted_unique_preds)}
    ranked_preds = np.array([pred_to_rank[val] for val in y_pred], dtype=int)
    max_rank = len(sorted_unique_preds)

    sort_indices = np.argsort(y_true, kind='stable')
    
    groups = defaultdict(list)
    for i in sort_indices:
        groups[y_true[i]].append(ranked_preds[i])

    ft = FenwickTree(max_rank)
    concordant_pairs = 0.0
    
    for r_val, p_rank_list in sorted(groups.items()):
        for p_rank in p_rank_list:
            concordant_pairs += ft.query(p_rank - 1)
            concordant_pairs += 0.5 * ft.range_query(p_rank, p_rank)

        for p_rank in p_rank_list:
            ft.update(p_rank, 1.0)

    total_pairs = n * (n - 1) // 2
    for r_val, p_rank_list in groups.items():
        group_size = len(p_rank_list)
        total_pairs -= group_size * (group_size - 1) // 2

    if total_pairs <= 0:
        return np.nan  
        
    return concordant_pairs / total_pairs

def get_acc(y_pred, y_true, tol = 0.2):
    """
    ACC = fraction of samples where |y_pred - y_true|/y_true <= tol
    """
    return np.mean((np.abs(y_pred - y_true) / np.where(np.abs(y_true) < 1e-8, 1e-8, y_true)) <= tol)

def get_alpr(y_pred, y_true):
    '''
    ALPR = mean(|log2(y_pred/y_true)|)
    '''
    return np.mean(np.abs(np.log2(y_pred/y_true)))

