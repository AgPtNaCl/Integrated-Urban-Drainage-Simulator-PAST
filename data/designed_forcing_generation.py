from pyswmm import Simulation, Nodes
import pandas as pd
import torch
import h5py
import re
import os
import random
from tqdm import tqdm
import numpy as np

# designed rainfall in Beijing
def get_designed_forcing(event_length = 190, dt = 10):
    ts = np.arange(0, event_length, dt)
    return_year = np.random.uniform(0, 500)
    peak_coef = np.random.random()
    a = 1602.0 * (1.0 + 1.037 * np.log10(return_year)) * 6 * 1e-3  # [L/(s·hm^2)]
    b = 11.593
    c = 0.681
    sigma_1 = 1
    sigma_2 = 10
    peak_loc = int(peak_coef * len(ts))

    i1 = a * ((1.0 - c) * (ts[peak_loc] - ts[:peak_loc]) / peak_coef + b) / np.power((ts[peak_loc] - ts[:peak_loc]) / peak_coef + b, 1.0 + c)
    i2 = a * ((1.0 - c) * (ts[peak_loc:] - ts[peak_loc]) / (1.0 - peak_coef) + b) / np.power((ts[peak_loc:] - ts[peak_loc]) / (1.0 - peak_coef) + b, 1.0 + c)

    i = np.concatenate((i1, i2)) 
    i = i * dt
    i = np.round(i, 6) 
    # print(return_year, "Total rainfall [mm]: ", np.sum(i))
    rainfall = [round(x, 2) for x in i]
    width_range = (1, 10)  
    ratio = 0.87 * (1 + random.uniform(-0.5, 0.5))
    top_n = random.randint(2, 8)

    days = np.arange(1, event_length/dt+1)

    largest_rainfalls = np.partition(rainfall, -top_n)[-top_n:]

    avg_largest_rainfall = np.mean(largest_rainfalls)
    peak_height = avg_largest_rainfall * ratio
    peak_position = peak_loc + top_n
    width2 = np.random.uniform(width_range[0], width_range[1])

    inflow = peak_height * np.exp(-(days - peak_position)**2 / (2 * width2**2))

    return rainfall, inflow.tolist()

def modify_inp_file(original_inp_path, modified_inp_path, rainfall_data, inflow_data):
    with open(original_inp_path, 'r') as file:
        content = file.read()

    rainfall_iter = iter(rainfall_data)
    node1_inflow_iter = iter(inflow_data)

    def replace_rainfall_data(match):
        new_value = next(rainfall_iter, '0')  
        return f"{match.group(1)}{new_value}"

    def replace_node1_data(match):
        new_value = next(node1_inflow_iter, '0') 
        return f"{match.group(1)}{new_value}"

    modified_content = re.sub(r'(3h1yearpoint\s+\d{1,2}/\d{1,2}/\d{4}\s+\d{2}:\d{2}\s+)[\d.]+', replace_node1_data, content)
    modified_content = re.sub(r'(3h1yearRain\s+\d{1,2}/\d{1,2}/\d{4}\s+\d{2}:\d{2}\s+)[\d.]+', replace_rainfall_data, modified_content)

    with open(modified_inp_path, 'w') as modified_file:
        modified_file.write(modified_content)
    
    return rainfall_data, inflow_data


def run_simulation_and_extract_data(inp_file_path):
    with Simulation(inp_file_path) as sim:
        node_names = [node.nodeid for node in Nodes(sim)] 
        depth_data = {node_name: [] for node_name in node_names}
        flow_data = {node_name: [] for node_name in node_names}
        lateral_inflow_data = {node_name: [] for node_name in node_names}
        total_simulation_duration = sim.end_time - sim.start_time
        step_interval = total_simulation_duration / 19
        next_step_time = sim.start_time

        for step in sim:
            if sim.current_time >= next_step_time and next_step_time!=sim.end_time:
                for node_name in node_names:
                    node = Nodes(sim)[node_name]
                    depth_data[node_name].append(node.depth)
                    flow_data[node_name].append(node.total_inflow)
                    lateral_inflow_data[node_name].append(node.lateral_inflow)
                next_step_time += step_interval

    depth_df = pd.DataFrame.from_dict(depth_data, orient='index').transpose()
    flow_df = pd.DataFrame.from_dict(flow_data, orient='index').transpose()
    lateral_df = pd.DataFrame.from_dict(lateral_inflow_data, orient='index').transpose()
    return torch.tensor(depth_df.values.T, dtype=torch.float32).view(-1, 19, 1).transpose(0, 1),torch.tensor(flow_df.values.T, dtype=torch.float32).view(-1, 19, 1).transpose(0, 1),torch.tensor(lateral_df.values.T, dtype=torch.float32).view(-1, 19, 1).transpose(0, 1)

def generate_data_set(original_inp_path, output_file_path, num_samples):
    with h5py.File(output_file_path, 'w') as h5f:
        for i in tqdm(range(num_samples)):
            modified_inp_path =f"modified_{i}.inp"
            modified_out_path = f"modified_{i}.out"
            modified_rpt_path = f"modified_{i}.rpt"
            rainfall_data,inflow_data = get_designed_forcing()
            modify_inp_file(original_inp_path, modified_inp_path,rainfall_data,inflow_data)
            node_depth_data,node_flow_data,lateral_inflow_data = run_simulation_and_extract_data(modified_inp_path)
            
            h5f.create_dataset(f'rainfall_{i}', data=rainfall_data)
            h5f.create_dataset(f'inflow_{i}', data=inflow_data)
            h5f.create_dataset(f'lateral_inflow_{i}', data=lateral_inflow_data.numpy())
            h5f.create_dataset(f'depth_data_{i}', data=node_depth_data.numpy())
            h5f.create_dataset(f'flow_data_{i}', data=node_flow_data.numpy())

            os.remove(modified_inp_path)
            os.remove(modified_out_path)
            os.remove(modified_rpt_path)
    print("Done")

# 示例使用
original_inp_path = 'community.inp'
output_file_path = 'training_set.h5'
num_samples = 3000  # 指定生成样本的数量
generate_data_set(original_inp_path, output_file_path, num_samples)
