import numpy as np
import torch

def adj_matrixs(args, distance_matrix, dpcs_id, spcs_id, grid2adj_comp, grid2adj_coop, operators):
    dist_eps = args.dist_eps
    distance_matrix = torch.from_numpy(distance_matrix).float().to(args.device)
    ### competition adj
    adj_comp = torch.zeros((args.N,args.N)).to(args.device)
    for src_idx in range(args.N):
        for dst_idx in range(args.N):
            node_dist = distance_matrix[src_idx,dst_idx]
            if((node_dist <= dist_eps) and (operators[src_idx] != operators[dst_idx])): # neighboring
                adj_comp[src_idx,dst_idx] = node_dist
    adj_comp = adj_comp.unsqueeze(dim=0)/args.dist_norm 

    ### cooperation adj
    adj_coop = torch.zeros((args.N,args.N)).to(args.device)
    for src_idx in range(args.N):
        for dst_idx in range(args.N):
            node_dist = distance_matrix[src_idx,dst_idx]
            if((node_dist <= dist_eps) and (operators[src_idx] == operators[dst_idx]) and (src_idx != dst_idx)): # neighboring
                adj_coop[src_idx,dst_idx] = node_dist
    adj_coop = adj_coop.unsqueeze(dim=0)/args.dist_norm 
    
    grid2adj_comp = [torch.from_numpy(adj).unsqueeze(dim=0).to(args.device)/args.dist_norm for adj in grid2adj_comp]
    grid2adj_coop = [torch.from_numpy(adj).unsqueeze(dim=0).to(args.device)/args.dist_norm for adj in grid2adj_coop]
    print("# of grid2adj:",len(grid2adj_comp), len(grid2adj_coop))
    
    return adj_comp, adj_coop, grid2adj_comp, grid2adj_coop, distance_matrix/args.dist_norm

def orthogonal_init(tensor, gain=1):
    '''
    Fills the input `Tensor` using the orthogonal initialization scheme from OpenAI
    Args:
        tensor: an n-dimensional `torch.Tensor`, where :math:`n \geq 2`
        gain: optional scaling factor
    Examples:
        >>> w = torch.empty(3, 5)
        >>> orthogonal_init(w)
    '''
    if tensor.ndimension() < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")

    rows = tensor.size(0)
    cols = tensor[0].numel()
    flattened = tensor.new(rows, cols).normal_(0, 1)

    if rows < cols:
        flattened.t_()

    # Compute the qr factorization
    u, s, v = ch.svd(flattened, some=True)
    if rows < cols:
        u.t_()
    q = u if tuple(u.shape) == (rows, cols) else v
    with ch.no_grad():
        tensor.view_as(q).copy_(q)
        tensor.mul_(gain)
    return tensor

class RunningStat(object):
    '''
    Keeps track of first and second moments (mean and variance)
    of a streaming time series.
     Taken from https://github.com/joschu/modular_rl
     Math in http://www.johndcook.com/blog/standard_deviation/
    '''
    def __init__(self, shape):
        self._n = np.int64(0)
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)
    def push(self, x):
        x = np.asarray(x)
        n_sample, n_feat = x.shape
        assert n_feat == self._M.shape[-1]
        self._n += n_sample
        oldM = self._M.copy()
        self._M[...] = oldM + (x - oldM).sum(axis=0)/self._n # mean
        self._S[...] = self._S + ((x - oldM)*(x - self._M)).sum(axis=0)/(self._n-1) # sum of (x-x^bar)^2
    @property
    def n(self):
        return self._n
    @property
    def mean(self):
        return self._M
    @property
    def var(self):
        return self._S
    @property
    def std(self):
        return np.sqrt(self.var)
    @property
    def shape(self):
        return self._M.shape
    def reset(self):
        pass
        # self._n = 0
        # self._M = np.zeros(self._shape)
        # self._S = np.zeros(self._shape)

class Identity:
    '''
    A convenience class which simply implements __call__
    as the identity function
    '''
    def __call__(self, x, *args, **kwargs):
        return x

    def reset(self):
        pass

class ZFilter:
    """
    y = (x-mean)/std
    using running estimates of mean,std
    """
    def __init__(self, shape, center=True, scale=True, clip=None):
        assert shape is not None
        self.center = center
        self.scale = scale
        self.clip = clip
        self.rs = RunningStat(shape)
        # self.prev_filter = prev_filter

    def __call__(self, x, **kwargs):
        # x = self.prev_filter(x, **kwargs)
        self.rs.push(x)
        # print(self.rs.mean,self.rs.std)
        if self.center:
            x = x - self.rs.mean
        if self.scale:
            if self.center:
                x = x / (self.rs.std + 1e-8)
            else:
                diff = x - self.rs.mean
                diff = diff/(self.rs.std + 1e-8)
                x = diff + self.rs.mean
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x

    def reset(self):
        # self.prev_filter.reset()
        self.rs.reset()

class RewardFilter:
    """
    "Incorrect" reward normalization [copied from OAI code]
    Incorrect in the sense that we 
    1. update return
    2. divide reward by std(return) *without* subtracting and adding back mean
    """
    def __init__(self, shape, gamma, clip=None):
        assert shape is not None
        self.gamma = gamma
        # self.prev_filter = prev_filter
        self.rs = RunningStat(shape)
        self.ret = np.zeros(shape)
        self.clip = clip

    def __call__(self, x, **kwargs):
        # x = self.prev_filter(x, **kwargs)
        self.ret = self.ret * self.gamma + x
        self.rs.push(self.ret)
        x = x / (self.rs.std + 1e-8)
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x
    
    def reset(self):
        self.ret = np.zeros_like(self.ret)
        # self.prev_filter.reset()
        self.rs.reset()
