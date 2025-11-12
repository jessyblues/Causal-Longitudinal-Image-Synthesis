import pandas as pd
import os
import torch.nn as nn
import torch
from sklearn.model_selection import train_test_split
import pdb
import numpy as np
import networkx as nx

def get_data_for_causal_discovery(csv_path):
    
    df_csv = csv_path
    df = pd.read_csv(df_csv)

    from sklearn.model_selection import train_test_split

    Subjects = df["Subject"].unique()

    train_subj, test_subj = train_test_split(Subjects, test_size=0.3, random_state=42)

    df_train = df_pair[df_pair["Subject"].isin(train_subj)].drop(columns=["Subject"])
    df_test  = df_pair[df_pair["Subject"].isin(test_subj)].drop(columns=["Subject"])
    
    return df_train, df_test

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,
                 hidden_layers=5, activation_fn=nn.ReLU):
        super(MLP, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(activation_fn())
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation_fn())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)
 
def convert_causal_graph(causal_graph_nx: nx.DiGraph):

    causal_graph = {}
    full_graph = {}

    for node in causal_graph_nx.nodes():
        parents = list(causal_graph_nx.predecessors(node))
        causal_graph[node] = parents
        full_graph[node] = [n for n in causal_graph_nx.nodes() if n != node]

    return causal_graph, full_graph

def train_mlp(df_train, causal_graph, full_graph, causal=True, epochs=1000, lr=0.001):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    models = {}
    input_dims = {}

    
    df_train, df_val = train_test_split(df_train, test_size=0.2, random_state=42)
    
    for node in ['GreyMatter', 'WholeBrain', 'SegVentricles']:
        t_1 = f"{node}_t1"
        if causal:
            parents = causal_graph[t_1]
        else:
            parents = full_graph[t_1]
        
        input_dims[t_1] = len(parents)
        model = MLP(input_dim=len(parents), hidden_dim=64, output_dim=1).to(device)
        models[t_1] = (model, parents)
    
    criterion = nn.L1Loss()
    optimizers = {t_1: torch.optim.Adam(model.parameters(), lr=lr) for t_1, (model, _) in models.items()}
    
    X_train = {t_1: torch.tensor(df_train[parents].values, dtype=torch.float32).to(device)
               for t_1, (_, parents) in models.items()}
    y_train = {t_1: torch.tensor(df_train[t_1].values, dtype=torch.float32).view(-1, 1).to(device)
               for t_1 in models.keys()}
    best_loss = {t_1: float('inf') for t_1 in models.keys()}
    best_models = {}
    for epoch in range(epochs):
        for t_1, (model, _) in models.items():
            model.train()
            optimizers[t_1].zero_grad()
            outputs = model(X_train[t_1])
            loss = criterion(outputs, y_train[t_1])
            loss.backward()
            optimizers[t_1].step()
            
        if (epoch+1) % 100 == 0:
            # 验证集评估
            for t_1, (model, _) in models.items():
                model.eval()
                with torch.no_grad():
                    X_val = torch.tensor(df_val[models[t_1][1]].values, dtype=torch.float32).to(device)
                    y_val = torch.tensor(df_val[t_1].values, dtype=torch.float32).view(-1, 1).to(device)
                    outputs = model(X_val)
                    val_loss = criterion(outputs, y_val)
                    
                    if val_loss.item() < best_loss[t_1]:
                        best_loss[t_1] = val_loss.item()
                        best_models[t_1] = model.state_dict()
            #print(f"Epoch [{epoch+1}/{epochs}] completed.")
            #print({t_1: round(best_loss[t_1], 4) for t_1 in models.keys()})
    
    for t_1, model in models.items():
        model[0].load_state_dict(best_models[t_1])         
    
    return models, best_loss



def test_mlp(models, df_test):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.L1Loss(reduction='none')
    
    X_test = {t_1: torch.tensor(df_test[parents].values, dtype=torch.float32).to(device)
              for t_1, (_, parents) in models.items()}
    y_test = {t_1: torch.tensor(df_test[t_1].values, dtype=torch.float32).view(-1, 1).to(device)
              for t_1 in models.keys()}
    
    test_losses = {}
    for t_1, (model, _) in models.items():
        model.eval()
        with torch.no_grad():
            outputs = model(X_test[t_1])/1000
            y_test[t_1] = y_test[t_1]/1000
            
            ## 计算每个样本的绝对误差
            losses = criterion(outputs, y_test[t_1])
            mean_loss = losses.mean().item()
            ## 计算rmse
            mse_loss = nn.MSELoss(reduction='none')
            losses = mse_loss(outputs, y_test[t_1])
            rmse_mean_loss = losses.mean().sqrt().item()
            ## 计算pearson相关系数
            outputs_np = outputs.cpu().numpy().flatten()
            y_test_np = y_test[t_1].cpu().numpy().flatten()
            if len(set(outputs_np)) > 1 and len(set(y_test_np)) > 1:
                corr_matrix = np.corrcoef(outputs_np, y_test_np)
                pearson_corr = corr_matrix[0, 1]
            else:
                pearson_corr = 0.0
            test_losses[t_1] = {'MAE': round(mean_loss, 2), 'Pearson': round(pearson_corr, 4), 'RMSE': round(rmse_mean_loss, 2)}
    
    return test_losses

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_csv', type=str, required=True, help='Path to the longitudinal data CSV file.')
    parser.add_argument('--causal_graph_path', type=str, required=True, help='The discovered causal graph path.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the trained models and results.')
    args = parser.parse_args()
    G = nx.read_graphml(args.causal_graph_path)

    df_train, df_test = get_data_for_causal_discovery(args.data_csv)
    causal_graph, full_graph = convert_causal_graph(G)
    causal_models, best_causal_val_loss = train_mlp(df_train, causal_graph=causal_graph, full_graph=full_graph, causal=True, epochs=5000, lr=0.001)
    
    for t_1, model in causal_models.items():
        torch.save(model[0].state_dict(), os.path.join(args.output_dir, f'causal_mlp_{t_1}.pth'))
