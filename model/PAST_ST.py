import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.nn import GATConv

class TGCNmodel(nn.Module):
    def __init__(self, num_nodes, lstm_hidden_size, edge_index, hid_dim_lstmrr=64, hid_dim_gat=24):
        super(TGCNmodel, self).__init__()
        self.num_nodes = num_nodes
        self.hid_dim_lstmrt = lstm_hidden_size
        self.hid_dim_lstmrr = hid_dim_lstmrr
        self.edge_index = edge_index
        self.hid_dim_gat = hid_dim_gat
        self.lstm = nn.LSTM(input_size=1,
                            hidden_size=self.hid_dim_lstmrr,
                            batch_first=True)
        self.fc = nn.Linear(self.hid_dim_lstmrr, self.num_nodes)
        self.conv1 = GATConv(in_channels=3, out_channels=3, heads=8, concat=True)
        self.conv2 = GATConv(in_channels=self.hid_dim_gat, out_channels=2, heads=8, concat=False)
        self.lstm_cell = nn.LSTMCell(self.num_nodes*2, self.hid_dim_lstmrt)
        self.ln = nn.LayerNorm(self.hid_dim_lstmrt)
        self.linear = nn.Linear(self.hid_dim_lstmrt, self.num_nodes*2) 
        for name, param in self.lstm_cell.named_parameters():
            if 'weight_ih' in name: 
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name: 
                nn.init.orthogonal_(param.data)
            elif 'bias' in name: 
                nn.init.constant_(param.data, 0)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, rainfall,inflow, device):
        batch_size = rainfall.shape[0]
        seq_len = rainfall.shape[1]
        edge_index = self.edge_index.to(device)
        edge_index = edge_index.view(2, 1, -1).repeat(1, batch_size, 1) + torch.arange(batch_size, device=device).view(1, -1, 1) * self.num_nodes
        edge_index = edge_index.view(2, -1)
        Graph_pred_list = []
        Lateral_list = []
        Attention_list = []
        h0 = torch.zeros(batch_size, self.hid_dim_lstmrt).to(device)
        hn = h0
        c0 = torch.zeros(batch_size, self.hid_dim_lstmrt).to(device)
        cn = c0
        xn = torch.zeros(batch_size, self.num_nodes, 2).to(device)
        h00 = torch.zeros(1, batch_size, self.hid_dim_lstmrr).to(device)
        c00 = torch.zeros(1, batch_size, self.hid_dim_lstmrr).to(device)
        lstm_out, _ = self.lstm(rainfall,(h00,c00))
        runoff = F.leaky_relu(self.fc(lstm_out)) 
        # process at each time step
        for t in range(seq_len):
            # concat hydrodynamic drivers
            current_runoff = runoff[:, t, :]
            current_runoff[:,753] = current_runoff[:,753] + inflow[:, t, 0] 
            lateral_inflow = current_runoff.unsqueeze(2) 
            x = torch.cat([xn, lateral_inflow], dim=-1) 
            x_gcn = x.contiguous()
            x_gcn = x_gcn.view(batch_size * self.num_nodes, -1)
            x_gcn, attention = self.conv1(x_gcn, edge_index, return_attention_weights = True)
            x_gcn = F.leaky_relu(x_gcn) 
            x_gcn = self.conv2(x_gcn, edge_index)
            x_gcn = F.leaky_relu(x_gcn) 
            x = x_gcn.reshape(batch_size, -1)  
            hn, cn = self.lstm_cell(x, (hn, cn)) 
            hn = self.ln(hn)
            xn = F.softplus(self.linear(hn))
            Graph_pred_list.append(xn)
            Lateral_list.append(lateral_inflow)
            Attention_list.append(attention)
            xn = xn.reshape(batch_size,self.num_nodes,-1)
        # integrate spatial-temporal diagram
        prediction = torch.stack(Graph_pred_list, dim=1) 
        Lateral = torch.stack(Lateral_list, dim=1)
        return  prediction, Lateral, Attention_list
