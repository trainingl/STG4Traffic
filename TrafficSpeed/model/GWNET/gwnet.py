import torch
import torch.nn as nn
import torch.nn.functional as F

class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        # x: (batch, channel, nodes, timesteps)
        x = torch.einsum('ncvl,vw->ncwl', (x, A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        # 使用1x1的卷积核替代Linear层
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        # x: (batch, channel, nodes, timesteps)
        return self.mlp(x)


class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(gcn, self).__init__()
        self.nconv = nconv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        """
        :param x: (batch, channel, nodes, timesteps)
        :param support: list of adjacent matrix
        """
        out = [x]
        # Multi-Graph
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            # MixHop: n-order
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        # Putting it together in the channel dimension
        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class gwnet(nn.Module):
    def __init__(self,
                 device,
                 num_nodes,
                 dropout=0.3,
                 supports=None,
                 gcn_bool=True,
                 addaptadj=True,
                 aptinit=None,
                 in_dim=2,
                 out_dim=12,
                 residual_channels=32,
                 dilation_channels=32,
                 skip_channels=256,
                 end_channels=512,
                 kernel_size=2,
                 blocks=4,
                 layers=2):
        # skip_channels = dilation_channels * (blocks * layers)
        super(gwnet, self).__init__()
        self.dropout = dropout
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj
        self.blocks = blocks
        self.layers = layers

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))

        # 1.list of adjacency matrix
        self.supports = supports
        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)
        else:
            self.supports = []
        if gcn_bool and addaptadj:
            if aptinit is None:
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
                self.supports_len += 1
            else:
                # ===================================================================
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                # ===================================================================
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(device)
                self.supports_len += 1

        # 2.Stacked Gated Temporal Convolutional Layers
        receptive_field = 1
        for b in range(blocks):
            # Here the convolution kernel is fixed.
            additional_scope = kernel_size - 1   # 1
            new_dilation = 1
            # Each layer requires padding = 3 data fills keeping the original length unchanged.
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1, kernel_size),
                                                   dilation=new_dilation))
                self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size),
                                                 dilation=new_dilation))
                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=skip_channels,
                                                     kernel_size=(1, 1)))
                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                # padding = (kernel_size - 1) * dilation
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                # Graph Convolution Network
                if self.gcn_bool:
                    self.gconv.append(gcn(c_in=dilation_channels,
                                          c_out=residual_channels,
                                          dropout=dropout,
                                          support_len=self.supports_len))

        # 3.Output prediction layer
        self.end_conv_1 = nn.Conv2d(
            in_channels=skip_channels,
            out_channels=end_channels,
            kernel_size=(1, 1),
            bias=True
        )  # channels from 256 to 512

        self.end_conv_2 = nn.Conv2d(
            in_channels=end_channels,
            out_channels=out_dim,
            kernel_size=(1, 1),
            bias=True
        )  # channels from 512 to 12

        self.receptive_field = receptive_field  # 1 + (4 * 3) = 13

    def forward(self, input):
        """
        Here one-dimensional convolutional kernels are used to extract temporal information,
        and the size of the convolutional kernels is constant at 2.
        :param input: (batch, in_channel, nodes, timesteps)
        :return:
        """
        input = input.permute(0, 3, 2, 1)
        in_len = input.size(3)   # timesteps
        if in_len < self.receptive_field:
            x = nn.functional.pad(input, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = input
        x = self.start_conv(x)
        skip = 0

        # calculate the adaptive adjacent matrix
        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]

        # WaveNet layers
        for i in range(self.blocks * self.layers):
            #            |----------------------------------------|   *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*
            residual = x
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate
            # the length of timesteps decreases after each temporal convolution.
            # skip connection
            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :, -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            # residual connnection
            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    x = self.gconv[i](x, new_supports)
                else:
                    x = self.gconv[i](x, self.supports)
            else:
                x = self.residual_convs[i](x)
            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)

        # skip: (batch, channel, nodes, 1)  It can be understood as aggregating temporal features.
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x


if __name__ == "__main__":
    x = torch.randn(64, 12, 207, 2)
    supports = [torch.randn(207, 207)]
    layer = gwnet(device='cpu',
                 num_nodes=207,
                 dropout=0.3,
                 supports=None,
                 gcn_bool=True,
                 addaptadj=True,
                 aptinit=None,
                 in_dim=2,
                 out_dim=12,
                 residual_channels=32,
                 dilation_channels=32,
                 skip_channels=256,
                 end_channels=512,
                 kernel_size=2,
                 blocks=4,
                 layers=2)
    y = layer(x.permute(0, 3, 2, 1))
    print(y.size())