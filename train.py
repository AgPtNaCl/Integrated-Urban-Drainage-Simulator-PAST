from dataset import *
from torch.utils.data import DataLoader
from Data_preprocess import *
from util import config
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.optim as optim
from model.PAST import TGCNmodel as PASTModel, Train
from model.LSTM import TGCNmodel as LSTMModel
from model.GAT import TGCNmodel as GATModel
from model.PAST_ST import TGCNmodel as PASTSTModel
from model.PAST_DR import TGCNmodel as PASTDRModel, Train_dr
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train PAST with optional config overrides.")
    # Model selection
    parser.add_argument("--model_name", type=str, choices=["PAST", "GAT", "LSTM", "PASTST", "PASTDR"], default="PAST",
                        help="Model architecture to use.")
    # Parameters
    parser.add_argument("--epochs", type=int, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, help="Initial learning rate.")
    parser.add_argument("--eta_min", type=float, help="Minimum learning rate for scheduler.")
    parser.add_argument("--lstm_hidden_size", type=int, help="Hidden size of LSTM.")
    parser.add_argument("--num_samples", type=int, help="Number of samples in training set.")
    parser.add_argument("--valid_num", type=int, help="Number of samples in validation set.")
    return parser.parse_args()


def override_config(args):
    """Override fields in config based on command line arguments"""
    if args.epochs is not None:
        config["training_params"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training_params"]["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        config["training_params"]["learning_rate"] = args.learning_rate
    if args.eta_min is not None:
        config["training_params"]["eta_min"] = args.eta_min
    if args.lstm_hidden_size is not None:
        config["training_params"]["lstm_hidden_size"] = args.lstm_hidden_size
    if args.num_samples is not None:
        config["training_params"]["num_samples"] = args.num_samples
    if args.valid_num is not None:
        config["validation_params"]["valid_num"] = args.valid_num


def main():
    # Parse and Override Config
    args = parse_args()
    override_config(args)

    # Configuration & Graph Info
    files = dict(config['filepath'])
    graph, num_nodes, edge_index = get_graph_info(files)

    # model_dict
    MODEL_FACTORY = {
    "PAST": PASTModel,
    "GAT": GATModel,
    "LSTM": LSTMModel,
    "PASTST":PASTSTModel,
    "PASTDR":PASTDRModel
    }

    # Training Parameters
    training_params = config['training_params']
    lstm_hidden_size = training_params['lstm_hidden_size']
    epochs = training_params['epochs']
    batch_size = training_params['batch_size']
    learning_rate = training_params['learning_rate']
    eta_min = training_params['eta_min']
    validation_set_size = config['validation_params']['valid_num']

    # Environment Setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(42)

    # Model Initialization
    model_class = MODEL_FACTORY[args.model_name]
    model = model_class(num_nodes, lstm_hidden_size, edge_index).to(device)
    print(f"Using model: {args.model_name}")

    # Data Loading
    train_dataset = Trainingset(config, num_nodes)
    if model_class == PASTDRModel:
        train_dataset = Trainingset_DR(config, num_nodes)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    valid_dataset = Validationset(config, num_nodes)
    if model_class == PASTDRModel:
        valid_dataset = Validationset_DR(config, num_nodes)
    valid_loader = DataLoader(valid_dataset, batch_size=validation_set_size)

    # Optimizer & Scheduler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)

    # Training
    if model_class == PASTDRModel:
        Train_dr(model, train_loader, valid_loader, epochs, optimizer, scheduler, device, save_path = f"checkpoints/{args.model_name.lower()}.pth")
    else:
        Train(model, train_loader, valid_loader, epochs, optimizer, scheduler, device, save_path = f"checkpoints/{args.model_name.lower()}.pth")

if __name__ == "__main__":
    main()