import torch
import torch.nn as nn
import torch.nn.functional as F


class cheb_conv(nn.Module):
    '''
        K-order chebyshev graph convolution
    '''
    def __init__(self, K, cheb_polynomials, in_channels, out_channels):
        '''
        :param K: int
        :param in_channles: int, num of channels in the input sequence
        :param out_channels: int, num of channels in the output sequence
        '''
        super(cheb_conv, self).__init__()
        self.K = K
        self.cheb_polynomials = cheb_polynomials
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DEVICE = cheb_polynomials[0].device
        self.Theta = nn.ParameterList([nn.Parameter(torch.FloatTensor(in_channels, out_channels).to(self.DEVICE)) for _ in range(K)])

    def forward(self, x):
        '''
        Chebyshev graph convolution operation
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, F_out, T)
        '''
        batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape
        outputs = []
        for time_step in range(num_of_timesteps):
            graph_signal = x[:, :, :, time_step]  # (b, N, F_in)
            output = torch.zeros(batch_size, num_of_vertices, self.out_channels).to(self.DEVICE)  # (b, N, F_out)
            for k in range(self.K):
                T_k = self.cheb_polynomials[k]  # (N,N)
                theta_k = self.Theta[k]  # (in_channel, out_channel)
                rhs = graph_signal.permute(0, 2, 1).matmul(T_k).permute(0, 2, 1)
                output = output + rhs.matmul(theta_k)
            outputs.append(output.unsqueeze(-1))
        return F.relu(torch.cat(outputs, dim=-1))


class MSTGCN_block(nn.Module):
    def __init__(self, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, cheb_polynomials):
        super(MSTGCN_block, self).__init__()
        self.cheb_conv = cheb_conv(K, cheb_polynomials, in_channels, nb_chev_filter)
        self.time_conv = nn.Conv2d(nb_chev_filter, nb_time_filter, kernel_size=(1, 3), stride=(1, time_strides), padding=(0, 1))
        self.residual_conv = nn.Conv2d(in_channels, nb_time_filter, kernel_size=(1, 1), stride=(1, time_strides))
        self.ln = nn.LayerNorm(nb_time_filter)

    def forward(self, x):
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, nb_time_filter, T)
        '''
        # cheb gcn
        spatial_gcn = self.cheb_conv(x)  # (b,N,F,T)
        # convolution along the time axis
        time_conv_output = self.time_conv(spatial_gcn.permute(0, 2, 1, 3))  # (b,F,N,T)
        # residual shortcut
        x_residual = self.residual_conv(x.permute(0, 2, 1, 3))  # (b,F,N,T)
        x_residual = self.ln(F.relu(x_residual + time_conv_output).permute(0, 3, 2, 1)).permute(0, 2, 3, 1)  # (b,N,F,T)
        return x_residual


class MSTGCN_submodule(nn.Module):
    def __init__(self, DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, cheb_polynomials, num_for_predict, len_input):
        '''
        :param nb_block:
        :param in_channels:
        :param K:
        :param nb_chev_filter:
        :param nb_time_filter:
        :param time_strides:
        :param cheb_polynomials:
        :param nb_predict_step:
        '''
        super(MSTGCN_submodule, self).__init__()
        self.BlockList = nn.ModuleList([MSTGCN_block(in_channels, K, nb_chev_filter, nb_time_filter, time_strides, cheb_polynomials)])
        self.BlockList.extend([MSTGCN_block(nb_time_filter, K, nb_chev_filter, nb_time_filter, 1, cheb_polynomials) for _ in range(nb_block-1)])
        self.final_conv = nn.Conv2d(int(len_input/time_strides), num_for_predict, kernel_size=(1, nb_time_filter))
        self.DEVICE = DEVICE
        self.to(DEVICE)

    def forward(self, x):
        '''
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, T_out)
        '''
        for block in self.BlockList:
            x = block(x)
        output = self.final_conv(x.permute(0, 3, 1, 2))[:, :, :, -1].permute(0, 2, 1)
        return output


class MSTGCN(nn.Module):
    """
    MSTGCN, 3 sub-modules, for hour, day, week respectively
    """
    def __init__(self,
                 DEVICE,
                 cheb_polynomials,
                 nb_block=2,
                 in_channels=3,
                 K=3,
                 nb_chev_filter=64,
                 nb_time_filter=64,
                 num_of_week=1,
                 num_of_day=1,
                 num_of_hour=3,
                 num_for_prediction=12,
                 points_per_hour=12):
        super(MSTGCN, self).__init__()
        if num_of_week <= 0 and num_of_hour <= 0 and num_of_hour <= 0:
            raise ValueError("The length of time steps must be greater than 0")
        self.submodules = nn.ModuleList([])
        time_strides = [num_of_week, num_of_day, num_of_hour]
        for time_stride in time_strides:
            if time_stride == 0:
                continue
            self.submodules.append(
                MSTGCN_submodule(
                    DEVICE=DEVICE,
                    nb_block=nb_block,
                    in_channels=in_channels,
                    K=K,
                    nb_chev_filter=nb_chev_filter,
                    nb_time_filter=nb_time_filter,
                    time_strides=time_stride,
                    cheb_polynomials=cheb_polynomials,
                    num_for_predict=num_for_prediction,
                    len_input=points_per_hour * time_stride
                )
            )

    def forward(self, x_list):
        """
        :param: x_list: list[mx.ndarray],
                shape is (batch_size, num_of_vertices, num_of_features, num_of_timesteps)
        :return: (batch_size, num_of_vertices, num_of_timesteps)
        """
        if len(x_list) != len(self.submodules):
            raise ValueError("num of submodule not equals to length of the input list")
        submodule_outputs = []
        for idx, submodule in enumerate(self.submodules):
            submodule_result = submodule(x_list[idx])
            submodule_result = torch.unsqueeze(submodule_result, dim=-1)
            submodule_outputs.append(submodule_result)

        submodule_outputs = torch.cat(submodule_outputs, dim=-1)
        return torch.sum(submodule_outputs, dim=-1)