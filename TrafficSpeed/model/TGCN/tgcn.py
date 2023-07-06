import torch
import torch.nn as nn

class TGCNGraphConv(nn.Module):
    def __init__(self, adj_mx, input_dim: int, hidden_dim: int, output_dim: int, bias: float = 0.0):
        """
        Args:
            1.input_dim: the feature dim of each node.
            2.hidden_dim: the feature dim of the hidden state.
            3.output_dim: the feature dim of the output.
        """
        super(TGCNGraphConv,self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.laplacian = self.get_laplacian_matrix(adj_mx)
        self.bias_init_value = bias
        self.weights = nn.Parameter(torch.FloatTensor(self.input_dim + self.hidden_dim, self.output_dim))
        self.biases = nn.Parameter(torch.FloatTensor(self.output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)
        nn.init.constant_(self.biases, self.bias_init_value)

    def get_laplacian_matrix(self, A):
        # A = A + I
        A = A + torch.eye(A.size(0), dtype=torch.float32).to(A.device)
        # degree matrix
        row_sum = torch.sum(A, axis=1, keepdim=False)  # (N,)
        d_inv_sqrt = torch.pow(row_sum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        degree_matrix = torch.diag(d_inv_sqrt)
        # D^(-1/2)AD^(-/12)
        laplacian = torch.matmul(torch.matmul(degree_matrix, A), degree_matrix)
        return laplacian

    def forward(self, inputs, hidden_state):
        """
        Args:
            inputs: the shape is (batch_size, num_nodes, feature_dim).
            hidden_state: the shape is (batch_size, num_nodes, hidden_dim).
        """
        # https://github.com/lehaifeng/T-GCN
        batch_size, num_nodes, input_dim = inputs.shape
        _, _, hidden_dim = hidden_state.shape
        # [x, h] -> concatenation: [batch_size, num_nodes, input_dim + hidden_dim]
        concatenation = torch.cat((inputs, hidden_state), dim=2)
        # Simplified writing: torch.einsum('nm, bmc->bnc', self.laplacian, concatenation)
        concatenation = concatenation.permute(1, 2, 0)
        concatenation = concatenation.reshape(
            (num_nodes, (hidden_dim + input_dim) * batch_size)
        )
        # A * [x, h]
        AX = torch.matmul(self.laplacian, concatenation)
        AX = AX.reshape(
            (num_nodes, (hidden_dim + input_dim), batch_size)
        )
        # A[x, h] (batch_size, num_nodes, hidden_dim + input_dim)
        AX = AX.permute(2, 0, 1)
        outputs = torch.matmul(AX, self.weights) + self.biases
        return outputs


class TGCNCell(nn.Module):
    def __init__(self, adj_mx, input_dim: int, hidden_dim: int):
        super(TGCNCell, self).__init__()
        self.hidden_dim = hidden_dim
        self.adj_mx = adj_mx
        # forget gate
        self.graph_conv1 = TGCNGraphConv(
            self.adj_mx, input_dim, self.hidden_dim, self.hidden_dim * 2, bias=1.0
        )
        # update gate
        self.graph_conv2 = TGCNGraphConv(
            self.adj_mx, input_dim, self.hidden_dim, self.hidden_dim
        )

    def forward(self, inputs, hidden_state):
        # [r, u] = sigmoid(A[x, h]W + b)
        concatenation = torch.sigmoid(self.graph_conv1(inputs, hidden_state))
        # r: [batch_size, num_nodes, hidden_dim]
        # u: [batch_size, num_nodes, hidden_dim]
        r, u = torch.chunk(concatenation, chunks=2, dim=-1)
        # c = tanh(A[x, (r * h)W + b])
        c = torch.tanh(self.graph_conv2(inputs, r * hidden_state))
        # h = u * h + (1 - u) * c
        new_hidden_state = u * hidden_state + (1.0 - u) * c
        # output_state, new_hidden_state
        return new_hidden_state, new_hidden_state

    
class TGCN(nn.Module):
    def __init__(self, adj_mx, input_dim: int, hidden_dim: int, out_dim: int):
        super(TGCN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.tgcn_cell = TGCNCell(adj_mx, input_dim, hidden_dim)
        self.regressor = nn.Linear(hidden_dim, out_dim)
    
    def forward(self, inputs):
        # inputs: (batch, timesteps, nodes, feature_dim)
        batch_size, seq_len, num_nodes, _ = inputs.shape
        hidden_state = torch.zeros(batch_size, num_nodes, self.hidden_dim).type_as(inputs)
        hidden_state.to(inputs.device)
        output_hidden = []
        for i in range(seq_len):
            output, hidden_state = self.tgcn_cell(inputs[:, i, :, :], hidden_state)
            output_hidden.append(output)
        # output_hidden = torch.stack(output_hidden, dim=1)
        # print(output_hidden.shape)
        last_hidden_state = hidden_state
        outputs = self.regressor(last_hidden_state)
        outputs = outputs.unsqueeze(dim=-1).permute(0, 2, 1, 3)
        return outputs

# if __name__ == '__main__':
#     x = torch.randn(64, 12, 207, 1)
#     adj_mx = torch.randn(207, 207)
#     model = TGCN(adj_mx=adj_mx, input_dim=1, hidden_dim=64, out_dim=12)
#     y = model(x)
#     print(y.size())