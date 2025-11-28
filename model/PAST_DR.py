import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.nn import GATConv
import time
from Data_preprocess import check_status, drop_connect
import copy
class TGCNmodel(nn.Module):
    def __init__(self, num_nodes, lstm_hidden_size, edge_index, hid_dim_gat=24):
        super(TGCNmodel, self).__init__()
        self.num_nodes = num_nodes
        self.hid_dim_lstmrt = lstm_hidden_size
        self.edge_index = edge_index
        self.hid_dim_gat = hid_dim_gat
        self.conv1 = GATConv(in_channels=4, out_channels=3, heads=8, concat=True)
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

    def forward(self, forcing_data, device):
        batch_size = forcing_data.shape[0]
        seq_len = forcing_data.shape[1]
        inflow = forcing_data[:, :, 753, 1]
        edge_index = self.edge_index.to(device)
        edge_index = edge_index.view(2, 1, -1).repeat(1, batch_size, 1) + torch.arange(batch_size, device=device).view(1, -1, 1) * self.num_nodes
        edge_index = edge_index.view(2, -1)
        Graph_pred_list = []
        Attention_list = []
        h0 = torch.zeros(batch_size, self.hid_dim_lstmrt).to(device)
        hn = h0
        c0 = torch.zeros(batch_size, self.hid_dim_lstmrt).to(device)
        cn = c0
        xn = torch.zeros(batch_size, self.num_nodes, 2).to(device)
        for t in range(seq_len):
            # Check gate status and remove orifice edges
            status = check_status(inflow[:, t], xn)
            revised_edge_index = drop_connect(edge_index, status)
            current_forcing = forcing_data[:, t, :, :]
            x = torch.cat([xn, current_forcing], dim=-1) 
            x_gcn = x.contiguous()
            x_gcn = x_gcn.view(batch_size * self.num_nodes, -1)
            x_gcn, attention = self.conv1(x_gcn, revised_edge_index, return_attention_weights = True)
            x_gcn = F.leaky_relu(x_gcn) 
            x_gcn = self.conv2(x_gcn, revised_edge_index)
            x_gcn = F.leaky_relu(x_gcn) 
            x = x_gcn.reshape(batch_size, -1)  
            hn, cn = self.lstm_cell(x, (hn, cn)) 
            hn = self.ln(hn)
            xn = F.softplus(self.linear(hn)) 
            Graph_pred_list.append(xn)
            Attention_list.append(attention)
            xn = xn.reshape(batch_size,self.num_nodes,-1)
        prediction = torch.stack(Graph_pred_list, dim=1)  
        return  prediction, Attention_list

def valid(model, val_loader, device):
    model.to(device)  # 确保模型在正确的设备上
    criterion = nn. L1Loss()
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for val_forcing_data, val_labels in val_loader:
            # 确保所有验证数据都移动到了设备上
            val_forcing_data = val_forcing_data.to(device).to(torch.float32)
            val_labels = val_labels.to(device).to(torch.float32)
            val_predictions, _ = model(val_forcing_data, device)
            val_loss = criterion(val_predictions, val_labels)
            total_val_loss += val_loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    return avg_val_loss
   

def Train_dr(model, train_loader, val_loader, epochs, optimizer, scheduler, device, save_path, patience=20, min_delta=1e-4):
    model.to(device)
    criterion = nn.L1Loss()
    epoch_losses = []
    val_losses = []  
    start_time = time.time()

    # Early Stopping
    best_val_loss = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0  

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for forcing_data, labels in train_loader:
            forcing_data = forcing_data.to(device).to(torch.float32)
            labels = labels.to(device).to(torch.float32)

            optimizer.zero_grad()
            predictions, _ = model(forcing_data, device)
            loss = criterion(predictions, labels)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        scheduler.step()

        avg_loss = total_loss / len(train_loader)
        epoch_losses.append(avg_loss)

        val_loss = valid(model, val_loader, device)
        val_losses.append(val_loss)  

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            print(f"Validation Loss after epoch {epoch+1}: {val_loss:.4f}")

                # ---- Early Stopping + Save Best ----
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
            # Save the best model
            torch.save(best_model_wts, save_path)
            print(f"  >>> Best model updated and saved at epoch {epoch+1}, val_loss={val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    end_time = time.time()
    total_training_time = end_time - start_time
    print(f"Total training time: {total_training_time:.2f} seconds")
    print(f"Best Validation Loss: {best_val_loss:.4f}")


