from __future__ import division
import torch
import torch.nn as nn
from torch.nn import init
import numbers
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import math


class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncwl,vw->ncvl',(x,A))
        return x.contiguous()

class temporalnconv(nn.Module):
    def __init__(self):
        super(temporalnconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncwl,vl->ncwv',(x,A))
        return x.contiguous()

class dy_nconv(nn.Module):
    def __init__(self):
        super(dy_nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,nvwl->ncwl',(x,A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out,bias=True):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=bias)

    def forward(self,x):
        return self.mlp(x)


class mixprop01(nn.Module):
    def __init__(self, c_in, c_out, gdep, order, dropout, alpha):
        super(mixprop01, self).__init__()
        self.nconv = nconv()
        self.mlp = linear((gdep + 1) * c_in, c_out)
        self.mlp3 = linear((gdep * order + 1) * c_in, c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha
        self.order = order

    def forward(self, x, adj):

        # higher-orderer graph
        alist = []
        for i in range(len(adj)):
            iadj = adj[i] + torch.eye(adj[i].size(0)).to(x.device)
            di = iadj.sum(1)
            ai = iadj / di.view(-1, 1)
            alist.append(ai)

        h = x
        out = [h]
        # higher-order fusion
        for j in range(len(alist)):
            a = alist[j]
            for i in range(self.gdep):
                h = self.alpha * x + (1 - self.alpha) * self.nconv(h, a)
                out.append(h)
        ho = torch.cat(out, dim=1)
        ho = self.mlp3(ho)

        # # # k-th order
        # a = alist[5]
        # for i in range(self.gdep):
        #     h = self.alpha * x + (1 - self.alpha) * self.nconv(h, a)
        #     out.append(h)
        # ho = torch.cat(out, dim=1)
        # ho = self.mlp(ho)

        return ho

class temporalGNN(nn.Module):
    def __init__(self,c_in,c_out,gdep,dropout,alpha):
        super(temporalGNN, self).__init__()
        self.temporalnconv = temporalnconv()
        self.mlp = linear((gdep+1)*c_in,c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self,x,adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device)

        # # strategy 1
        # d = adj.sum(1)
        # a = adj / d.view(-1, 1)

        # strategy 2
        adj1 = torch.relu(adj)
        d1 = adj1.sum(1)
        one = torch.ones_like(d1)
        d1 = torch.where(d1 == 0., one, d1)
        a1 = adj1 / d1.view(-1, 1)
        adj2 = -torch.relu(-adj)
        d2 = adj2.sum(1)
        d2 = torch.where(d2 == 0., one, d2)
        a2 = adj2 / d2.view(-1, 1)
        a = a1 - a2

        h = x
        out = [h]
        for i in range(self.gdep):
            h = self.alpha*x + (1-self.alpha)*self.temporalnconv(h,a)
            out.append(h)
        ho = torch.cat(out,dim=1)
        ho = self.mlp(ho)
        return ho

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.temporalnconv = temporalnconv()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        output = self.temporalnconv(input, adj)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class AdaptiveGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(AdaptiveGCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

class dilated_1D(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(dilated_1D, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [2,3,6,7]
        self.tconv = nn.Conv2d(cin,cout,(1,7),dilation=(1,dilation_factor))

    def forward(self,input):
        x = self.tconv(input)
        return x

class dilated_inception(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(dilated_inception, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [2,3,6,7]
        cout = int(cout/len(self.kernel_set))
        for kern in self.kernel_set:
            self.tconv.append(nn.Conv2d(cin,cout,(1,kern),dilation=(1,dilation_factor)))

    def forward(self,input):
        x = []
        for i in range(len(self.kernel_set)):
            x.append(self.tconv[i](input))
        for i in range(len(self.kernel_set)):
            x[i] = x[i][...,-x[-1].size(3):]
        x = torch.cat(x,dim=1)
        return x

class graph_constructor2(nn.Module):
    def __init__(self, nnodes, predefined_A, dim, device, alpha=3, static_feat=None, order=3):
        super(graph_constructor2, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim,dim)
            self.lin2 = nn.Linear(dim,dim)

        self.device = device
        self.predefined_A = predefined_A

        self.ones = torch.ones_like(self.predefined_A)
        self.adjacentA = torch.where(self.predefined_A > 0., self.ones, self.predefined_A)

        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat
        self.order = order

    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0))

        adj = F.relu(torch.tanh(self.alpha*a))

        InnerAdjacentA = self.adjacentA.to(self.device)

        adj01list = [InnerAdjacentA]
        for i in range(self.order - 1):
            adj01i = torch.mm(adj01list[i], InnerAdjacentA)
            adj01i = torch.clamp(adj01i, max=1.0)
            diagi = torch.diag(torch.diag(adj01i))
            adj01i = adj01i - diagi
            for jadj in adj01list:
                adj01i = adj01i - jadj
            adj01i = torch.clamp(adj01i, min=0.0)
            adj01list.append(adj01i)

        adjlist = []
        for i in range(len(adj01list)):
            adjlist.append(adj * adj01list[i])
        return adjlist

class temporal_graph_constructor(nn.Module):
    def __init__(self, neiaccount, nnodes, dim, device, alpha=3, static_feat=None):
        super(temporal_graph_constructor, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim,dim)
            self.lin2 = nn.Linear(dim,dim)

        self.device = device
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat
        self.nnodes = nnodes
        self.neiaccount = neiaccount

    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1,0))

        adj = torch.tanh(self.alpha * a)

        # mask matrix
        mask = np.eye(self.nnodes, k=0)
        # contextual neighbors fusion
        for i in range(1, self.neiaccount + 1):
            mask = mask + np.eye(self.nnodes, k=i) + np.eye(self.nnodes, k=-i)

        # # k-th neighbor
        # mask = mask + np.eye(self.nnodes, k=self.neiaccount) + np.eye(self.nnodes, k=-self.neiaccount)

        mask = torch.from_numpy(mask).to(torch.float32).to(self.device)

        # # mask matrix2
        # mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        # mask.fill_(float('0'))
        # s1, t1 = adj.topk(self.k, 1)
        # mask.scatter_(1, t1, s1.fill_(1))

        adj = adj * mask

        return adj

class LayerNorm(nn.Module):
    __constants__ = ['normalized_shape', 'weight', 'bias', 'eps', 'elementwise_affine']
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
            self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input, idx):
        if self.elementwise_affine:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight[:,idx,:], self.bias[:,idx,:], self.eps)
        else:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight, self.bias, self.eps)

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)

