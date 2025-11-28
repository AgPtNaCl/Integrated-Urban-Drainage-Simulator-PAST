import os
import h5py
import torch
import random
import shapefile
import numpy as np
import networkx as nx
from torch_geometric.seed import seed_everything

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    seed_everything(seed)


def get_graph_info(files):
    G = nx.DiGraph()

    for feature_type, filepath in files.items():
        sf = shapefile.Reader(filepath)

        if feature_type =='Junctions':
            for shapeRecord in sf.shapeRecords():
                node_name = shapeRecord.record[0]
                node_x = shapeRecord.record[1]
                node_y = shapeRecord.record[2]
                node_invertelev = shapeRecord.record[7]
                initdepth = shapeRecord.record[10]

                G.add_node(node_name, pos=(node_x, node_y),initdepth = initdepth,invertelev=node_invertelev)
        
        elif feature_type in ['Outfalls', 'Storages']:
            for shapeRecord in sf.shapeRecords():
                node_name = shapeRecord.record[0]
                node_x = shapeRecord.record[1]
                node_y = shapeRecord.record[2]
                node_invertelev = shapeRecord.record[7]
                initdepth = 0
                G.add_node(node_name, pos=(node_x, node_y),initdepth = initdepth,invertelev=node_invertelev)

        elif feature_type == 'Conduits':
            for shapeRecord in sf.shapeRecords():
                edge_name = shapeRecord.record[0]
                edge_start = shapeRecord.record[1]
                edge_end = shapeRecord.record[2]
                G.add_edge(edge_start, edge_end, name=edge_name)

        elif feature_type == 'Pumps':
            for shapeRecord in sf.shapeRecords():
                edge_name = shapeRecord.record[0]
                edge_start = shapeRecord.record[1]
                edge_end = shapeRecord.record[2] 
                G.add_edge(edge_start, edge_end, name=edge_name)

        elif feature_type == 'Orifices':
            for shapeRecord in sf.shapeRecords():
                edge_name = shapeRecord.record[0]
                edge_start = shapeRecord.record[1]
                edge_end = shapeRecord.record[2]
                G.add_edge(edge_start, edge_end, name=edge_name)

        elif feature_type == 'Weirs':
            for shapeRecord in sf.shapeRecords():
                edge_name = shapeRecord.record[0]
                edge_start = shapeRecord.record[1]
                edge_end = shapeRecord.record[2]
                G.add_edge(edge_start, edge_end, name=edge_name)
        
        elif feature_type == 'Outlets':
            for shapeRecord in sf.shapeRecords():
                edge_name = shapeRecord.record[0]
                edge_start = shapeRecord.record[1]
                edge_end = shapeRecord.record[2]
                G.add_edge(edge_start, edge_end, name=edge_name)


    nodes = list(G.nodes)
    edges = list(G.edges)
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    # directed graph
    edge_index = torch.tensor([[node_to_idx[edge[0]], node_to_idx[edge[1]]] for edge in edges], dtype=torch.long).t().contiguous()

    num_nodes=G.number_of_nodes()
    return G, num_nodes, edge_index

def drop_connect(edge_index, status):
    num_edges = edge_index.size(1)

    indices_to_keep = torch.ones(num_edges, dtype=torch.bool, device=edge_index.device)
    
    indices_to_drop = torch.nonzero(status == 0, as_tuple=False).flatten() * 1902
    indices_to_drop = indices_to_drop.unsqueeze(1) + torch.tensor([1898, 1899], device=edge_index.device) # 1898 and 1899 are the numbers of the edge corresponding to Orifices
    indices_to_drop = indices_to_drop.view(-1)
    
    indices_to_keep[indices_to_drop] = False

    result = edge_index[:, indices_to_keep]
    return result


def check_status(inflow, hydraulic_elements):
    condition = (inflow > 9) & (hydraulic_elements[:, 1361, 0] > 0.3)
    status = condition.to(torch.int)
    return status


def Merge_forcing_data(rainfall, inflow, num_nodes):
    batch_size, seq_len = rainfall.size() 
    rainfall = rainfall.view(batch_size, seq_len, 1).repeat(1, 1, num_nodes)
    inflow = inflow.view(batch_size, seq_len, 1)
    inflow_expanded = torch.zeros(batch_size, seq_len, num_nodes)
    inflow_expanded[:, :, 753] = inflow[:, :, 0]

    forcing_data=torch.cat((rainfall.unsqueeze(3),inflow_expanded.unsqueeze(3)),dim=-1)
    return forcing_data

def load_data_from_hdf5(file_path, num_samples, num_nodes):
    with h5py.File(file_path, 'r') as h5_file:
        rainfall_train=[]
        inflow_train = []
        node_depth_train = []
        node_flow_train = []

        for i in range(num_samples):
            rainfall_data =h5_file[f'rainfall_{i}'][()]
            inflow_data = h5_file[f'inflow_{i}'][()]
            node_depth_data = h5_file[f'depth_data_{i}'][()]
            node_flow_data = h5_file[f'flow_data_{i}'][()]

            rainfall_train.append(torch.tensor(rainfall_data, dtype=torch.float32))
            inflow_train.append(torch.tensor(inflow_data, dtype=torch.float32))
            node_depth_train.append(torch.tensor(node_depth_data, dtype=torch.float32))
            node_flow_train.append(torch.tensor(node_flow_data, dtype=torch.float32))

        rainfall_train = torch.stack(rainfall_train).view(num_samples, len(rainfall_data))  
        inflow_train = torch.stack(inflow_train).view(num_samples, len(inflow_data))
        node_depth_train = torch.stack(node_depth_train).view(num_samples, -1, num_nodes)
        node_flow_train = torch.stack(node_flow_train).view(num_samples, -1, num_nodes)

        labels=torch.cat([node_depth_train,node_flow_train],dim=-1)

    return rainfall_train, inflow_train, labels

def get_lateral_inflow(file_path, num_samples, num_nodes):
    with h5py.File(file_path, 'r') as h5_file:
        lateral_inflow_train = []
        node_depth_train = []
        node_flow_train = []

        for i in range(num_samples):

            lateral_inflow_data =h5_file[f'lateral_inflow_{i}'][()]
            node_depth_data = h5_file[f'depth_data_{i}'][()]
            node_flow_data = h5_file[f'flow_data_{i}'][()]
            lateral_inflow_train.append(torch.tensor(lateral_inflow_data, dtype=torch.float32))
            node_depth_train.append(torch.tensor(node_depth_data, dtype=torch.float32))
            node_flow_train.append(torch.tensor(node_flow_data, dtype=torch.float32))

        lateral_inflow_train = torch.stack(lateral_inflow_train)
        node_depth_train = torch.stack(node_depth_train).view(num_samples, -1, num_nodes)
        node_flow_train = torch.stack(node_flow_train).view(num_samples, -1, num_nodes)

        labels=torch.cat([node_depth_train,node_flow_train],dim=-1)

    return lateral_inflow_train, labels
