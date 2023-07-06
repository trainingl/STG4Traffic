import torch
import warnings
import torch.nn.functional as F
import torch.nn as nn


class AttLayer(nn.Module):
    def __init__(self, out_channels, use_bias=False, reduction=16):
        super(AttLayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(out_channels, out_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels // reduction, 1, bias=False),
            nn.Hardsigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, 1, 1)
        return x * y.expand_as(x)


class AVWGCN(nn.Module):
    def __init__(self, cheb_polynomials, L_tilde, dim_in, dim_out, cheb_k, embed_dim):
        super(AVWGCN, self).__init__()
        self.cheb_k = cheb_k
        self.cheb_polynomials = cheb_polynomials
        self.L_tilde = L_tilde
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))

        # for existing graph convolution
        # self.init_gconv = nn.Conv1d(dim_in, dim_out, kernel_size=5, padding=0)
        self.init_gconv = nn.Linear(dim_in, dim_out)
        self.gconv = nn.Linear(dim_out * cheb_k, dim_out)
        self.dy_gate1 = AttLayer(dim_out)
        self.dy_gate2 = AttLayer(dim_out)

    def forward(self, x, node_embeddings, L_tilde_learned):
        # x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        # output shape [B, N, C]
        b, n, _ = x.shape
        # 0) learned cheb_polynomials
        node_num = node_embeddings.shape[0]

        # L_tilde_learned = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
        # L_tilde_learned = torch.matmul(L_tilde_learned, self.L_tilde) * L_tilde_learned

        support_set = [torch.eye(node_num).to(L_tilde_learned.device), L_tilde_learned]
        # default cheb_k = 3
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * L_tilde_learned, support_set[-1]) - support_set[-2])

        # 1) convolution with learned graph convolution (implicit knowledge)
        supports = torch.stack(support_set, dim=0)
        weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)  # N, cheb_k, dim_in, dim_out
        bias = torch.matmul(node_embeddings, self.bias_pool)  # N, dim_out
        x_g = torch.einsum("knm,bmc->bknc", supports, x)  # B, cheb_k, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        x_gconv0 = torch.einsum('bnki,nkio->bno', x_g, weights) + bias  # b, N, dim_out

        # 2) convolution with existing graph (explicit knowledge)
        graph_supports = torch.stack(self.cheb_polynomials, dim=0)  # [k, n, m]
        x = self.init_gconv(x)
        x_g1 = torch.einsum("knm,bmc->bknc", graph_supports, x)
        x_g1 = x_g1.permute(0, 2, 1, 3).reshape(b, n, -1)  # B, N, cheb_k, dim_in
        x_gconv1 = self.gconv(x_g1)

        # 3) fusion of explit knowledge and implicit knowledge
        x_gconv = self.dy_gate1(F.leaky_relu(x_gconv0).transpose(1, 2)) + self.dy_gate2(
            F.leaky_relu(x_gconv1).transpose(1, 2))
        # x_gconv = F.leaky_relu(x_gconv0) + F.leaky_relu(x_gconv1)

        return x_gconv.transpose(1, 2)


class RGSLCell(nn.Module):
    def __init__(self, cheb_polynomials, L_tilde, node_num, dim_in, dim_out, cheb_k, embed_dim):
        super(RGSLCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = AVWGCN(cheb_polynomials, L_tilde, dim_in+self.hidden_dim, 2*dim_out, cheb_k, embed_dim)
        self.update = AVWGCN(cheb_polynomials, L_tilde, dim_in+self.hidden_dim, dim_out, cheb_k, embed_dim)

    def forward(self, x, state, node_embeddings, learned_tilde):
        #x: B, num_nodes, input_dim
        #state: B, num_nodes, hidden_dim
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings, learned_tilde))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z*state), dim=-1)
        hc = torch.tanh(self.update(candidate, node_embeddings, learned_tilde))
        h = r*state + (1-r)*hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)


def gumbel_softmax(logits: torch.Tensor, tau: float = 1, hard: bool = False, eps: float = 1e-10, dim: int = -1) -> torch.Tensor:
    if eps != 1e-10:
        warnings.warn("`eps` parameter is deprecated and has no effect.")

    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    # y_soft = gumbels.softmax(dim)
    y_soft = gumbels

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret


class AVWDCRNN(nn.Module):
    def __init__(self, cheb_polynomials, L_tilde, node_num, dim_in, dim_out, cheb_k, embed_dim, num_layers=1):
        super(AVWDCRNN, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(RGSLCell(cheb_polynomials, L_tilde, node_num, dim_in, dim_out, cheb_k, embed_dim))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(RGSLCell(cheb_polynomials, L_tilde, node_num, dim_out, dim_out, cheb_k, embed_dim))

    def forward(self, x, init_state, node_embeddings, learned_tilde):
        #shape of x: (B, T, N, D)
        #shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, node_embeddings, learned_tilde)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        #current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        #output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        #last_state: (B, N, hidden_dim)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)      #(num_layers, B, N, hidden_dim)


class RGSL(nn.Module):
    def __init__(self,
                 num_nodes,
                 input_dim,
                 rnn_units,
                 embed_dim,
                 output_dim,
                 horizon,
                 cheb_k,
                 num_layers,
                 default_graph,
                 cheb_polynomials,
                 L_tilde):
        super(RGSL, self).__init__()
        self.num_node = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = rnn_units
        self.output_dim = output_dim
        self.horizon = horizon
        self.num_layers = num_layers

        self.default_graph = default_graph
        self.node_embeddings = nn.Parameter(torch.randn(self.num_node, embed_dim), requires_grad=True)

        self.encoder = AVWDCRNN(cheb_polynomials, L_tilde, num_nodes, input_dim, rnn_units, cheb_k,
                                embed_dim, num_layers)

        # predictor
        self.end_conv = nn.Conv2d(1, horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)

        self.adj = None
        self.tilde = None

    def scaled_laplacian(self, node_embeddings, is_eval=False):
        # Normalized graph Laplacian function.
        # :param W: np.ndarray, [n_route, n_route], weighted adjacency matrix of G.
        # :return: np.matrix, [n_route, n_route].
        # learned graph
        node_num = self.num_node
        learned_graph = torch.mm(node_embeddings, node_embeddings.transpose(0, 1))
        norm = torch.norm(node_embeddings, p=2, dim=1, keepdim=True)
        norm = torch.mm(norm, norm.transpose(0, 1))
        learned_graph = learned_graph / norm
        learned_graph = (learned_graph + 1) / 2.
        # learned_graph = F.sigmoid(learned_graph)
        learned_graph = torch.stack([learned_graph, 1 - learned_graph], dim=-1)

        # make the adj sparse
        if is_eval:
            adj = gumbel_softmax(learned_graph, tau=1, hard=True)
        else:
            adj = gumbel_softmax(learned_graph, tau=1, hard=True)
        adj = adj[:, :, 0].clone().reshape(node_num, -1)
        # mask = torch.eye(self.num_nodes, self.num_nodes).to(device).byte()
        mask = torch.eye(node_num, node_num).bool().cuda()
        adj.masked_fill_(mask, 0)

        # d ->  diagonal degree matrix
        W = adj
        n = W.shape[0]
        d = torch.sum(W, axis=1)
        ## L -> graph Laplacian
        L = -W
        L[range(len(L)), range(len(L))] = d
        try:
            lambda_max = (L.max() - L.min())
        except Exception as e:
            print("eig error!!: {}".format(e))
            lambda_max = 1.0

        # pesudo laplacian matrix, lambda_max = eigs(L.cpu().detach().numpy(), k=1, which='LR')[0][0].real
        tilde = (2 * L / lambda_max - torch.eye(n).cuda())
        self.adj = adj
        self.tilde = tilde
        return adj, tilde

    def forward(self, x):
        # x: B, T, N, D
        if self.train:
            adj, learned_tilde = self.scaled_laplacian(self.node_embeddings, is_eval=False)
        else:
            adj, learned_tilde = self.scaled_laplacian(self.node_embeddings, is_eval=True)

        init_state = self.encoder.init_hidden(x.shape[0])
        output, _ = self.encoder(x, init_state, self.node_embeddings, learned_tilde)  # B, T, N, hidden
        output = output[:, -1:, :, :]  # B, 1, N, hidden

        # CNN based predictor
        output = self.end_conv((output))  # B, T*C, N, 1
        output = output.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.num_node)
        output = output.permute(0, 1, 3, 2)  # B, T, N, C
        return output