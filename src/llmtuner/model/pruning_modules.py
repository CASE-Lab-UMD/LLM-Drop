import torch
from torch import nn


class ExpertLinear(nn.Linear):
    """🔍 This is for pruning. (forward with scores for hooks to capture)"""

    def forward(self, input: torch.Tensor, routing_scores: torch.Tensor) -> torch.Tensor:
        return super().forward(input)


class GateLinear(nn.Linear):
    """🔍 This is for module locating."""


class LoSparseLinear(nn.Module):
    """
    Examples
    --------
    >>> import torch
    >>> linear = LoSparseLinear(16, 32, 2, has_bias=False)
    >>> inp = torch.randn(2, 16)
    >>> out = linear(inp)
    >>> out.shape
    torch.Size([2, 32])
    """

    def __init__(self, in_feature, out_feature, reduced_rank, has_bias=True, has_sparse=True):
        super().__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.reduced_rank = reduced_rank
        self.has_bias = has_bias
        self.has_sparse = has_sparse

        self.right = nn.Linear(in_feature, reduced_rank, bias=False)
        self.left = nn.Linear(reduced_rank, out_feature, bias=False)
        if self.has_sparse:
            self.sparse = nn.Linear(in_feature, out_feature, bias=False)

        if self.has_bias:
            self.bias = nn.Parameter(torch.zeros(out_feature, requires_grad=True))

        self.nonzero_idx = None
        self.sparse_weight_pruned = None
        self.SX = None
        self.SX_deberta = None  # Deberta will use Q and K again

    @property
    def weight(self):
        return self.left.weight @ self.right.weight

    def forward(self, x):
        """ Y = XW.T+B = X(LR+S).T+B = X(LR).T+XS.T+B """
        l_r_x = self.left(self.right(x))
        if self.has_sparse:
            if self.sparse_weight_pruned is not None:
                s_x_ = torch.matmul(x, self.sparse_weight_pruned.T)
                b_, l_, d_ = x.shape

                # restore y
                # keep record for the first forward
                if self.SX is None or self.SX_deberta is None:  # For QKV at the first time
                    out_feature, in_feature = self.sparse.weight.shape
                    device = x.device
                    if b_ != 1:
                        self.SX = torch.zeros(b_, l_, out_feature, device=device)
                        self.SX[..., self.nonzero_idx] = s_x_
                        y_ = l_r_x + self.SX + self.bias if self.has_bias else l_r_x + self.SX
                    else:  # For QK at the second time
                        self.SX_deberta = torch.zeros(b_, l_, out_feature, device=device)
                        self.SX_deberta[..., self.nonzero_idx] = s_x_
                        y_ = l_r_x + self.SX_deberta + self.bias if self.has_bias else l_r_x + self.SX_deberta

                # do not need to create new cuda memory
                else:
                    if b_ != 1:
                        self.SX[..., self.nonzero_idx] = s_x_
                        y_ = l_r_x + self.SX + self.bias if self.has_bias else l_r_x + self.SX
                    else:
                        self.SX_deberta[..., self.nonzero_idx] = s_x_
                        y_ = l_r_x + self.SX_deberta + self.bias if self.has_bias else l_r_x + self.SX_deberta
            else:
                s_x = self.sparse(x)
                y_ = l_r_x + s_x + self.bias if self.has_bias else l_r_x + s_x
        else:
            y_ = l_r_x + self.bias if self.has_bias else l_r_x
        return y_

    def initialize_weight(self, left_weight, right_weight, sparse_weight=None, bias=None, requires_grad=False):
        self.left.weight.data = left_weight.data
        self.right.weight.data = right_weight.data
        if self.has_sparse:
            self.sparse.weight.data = sparse_weight.data

    def prune_sparse(self):
        self.nonzero_idx = torch.nonzero(self.sparse.weight.sum(dim=1)).flatten()
        self.sparse_weight_pruned = nn.Parameter(self.sparse.weight[self.nonzero_idx, :])
