from layer import *
import numpy as np
from scipy.sparse import coo_matrix

class HOMnet(nn.Module):
    def __init__(self, gcn_true, buildA_true, gcn_depth, num_nodes, device, predefined_A=None, static_feat=None,
                 dropout=0.3, subgraph_size=20, node_dim=40, dilation_exponential=1, conv_channels=32,
                 residual_channels=32, skip_channels=64, end_channels=128, seq_length=12, in_dim=2, out_dim=12,
                 layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True, order=2, neiaccount=2):
        super(HOMnet, self).__init__()
        self.gcn_true = gcn_true
        self.buildA_true = buildA_true
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.predefined_A = predefined_A
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.order = order
        self.neiaccount = neiaccount
        self.device = device
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))

        # higher order: predefined connections but unknown weights
        self.gc = graph_constructor2(num_nodes, self.predefined_A, node_dim, device, alpha=tanhalpha,
                                    static_feat=static_feat, order=self.order)

        self.temporalgc = temporal_graph_constructor(neiaccount=neiaccount, nnodes=seq_length, dim=4, device=device,
                                         alpha=tanhalpha, static_feat=static_feat)

        self.seq_length = seq_length

        kernel_size = 7
        if dilation_exponential>1:
            self.receptive_field = int(1+(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
        else:
            self.receptive_field = layers*(kernel_size-1) + 1


        for j in range(1,layers+1):
            self.filter_convs.append(
                temporalGNN(residual_channels, conv_channels, gcn_depth, dropout, propalpha))
            self.gate_convs.append(
                temporalGNN(residual_channels, conv_channels, gcn_depth, dropout, propalpha))

            self.residual_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                out_channels=residual_channels,
                                             kernel_size=(1, 1)))
            if self.seq_length>self.receptive_field:
                self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                out_channels=skip_channels,
                                                kernel_size=(1, self.seq_length)))
            else:
                self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                out_channels=skip_channels,
                                                kernel_size=(1, self.seq_length)))

            if self.gcn_true:
                self.gconv1.append(mixprop01(conv_channels, residual_channels, gcn_depth, order, dropout, propalpha))
                self.gconv2.append(mixprop01(conv_channels, residual_channels, gcn_depth, order, dropout, propalpha))

            if self.seq_length>self.receptive_field:
                self.norm.append(LayerNorm((residual_channels, num_nodes, self.seq_length),elementwise_affine=layer_norm_affline))
            else:
                self.norm.append(LayerNorm((residual_channels, num_nodes, self.seq_length),elementwise_affine=layer_norm_affline))

        self.layers = layers
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                             out_channels=end_channels,
                                             kernel_size=(1,1),
                                             bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                             out_channels=out_dim,
                                             kernel_size=(1,1),
                                             bias=True)
        if self.seq_length > self.receptive_field:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.seq_length), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, self.seq_length), bias=True)

        else:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.seq_length), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, self.seq_length), bias=True)


        self.idx = torch.arange(self.num_nodes).to(device)
        self.temporalidx = torch.arange(self.seq_length).to(device)



    def forward(self, input, idx=None):
        seq_len = input.size(3)
        assert seq_len==self.seq_length, 'input sequence length not equal to preset sequence length'

        if self.gcn_true:
            if self.buildA_true:
                if idx is None:
                    adp = self.gc(self.idx)
                else:
                    adp = self.gc(idx)
            else:
                adp = self.predefined_A

        temporaladp = self.temporalgc(self.temporalidx)

        x = self.start_conv(input)
        skip = self.skip0(F.dropout(input, self.dropout, training=self.training))
        for i in range(self.layers):
            residual = x
            filter = self.filter_convs[i](x, temporaladp)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x, temporaladp)
            gate = torch.sigmoid(gate)
            x = filter * gate
            x = F.dropout(x, self.dropout, training=self.training)
            s = x
            s = self.skip_convs[i](s)
            skip = s + skip
            if self.gcn_true:
                transposeadp = []
                for iadp in adp:
                    transposeadp.append(iadp.transpose(1,0))
                x = self.gconv1[i](x, adp)+self.gconv2[i](x, transposeadp)

            else:
                # waiting revising
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]
            if idx is None:
                x = self.norm[i](x,self.idx)
            else:
                x = self.norm[i](x,idx)

        skip = self.skipE(x) + skip
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x
