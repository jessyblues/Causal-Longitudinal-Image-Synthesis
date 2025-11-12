import os
import torch
import pandas as pd
import networkx as nx
import torch.nn as nn
from tqdm import tqdm


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
    for node in causal_graph_nx.nodes():
        parents = list(causal_graph_nx.predecessors(node))
        causal_graph[node] = parents
    return causal_graph


def load_models(model_dir, causal_graph, device):
    models = {}
    for t_1 in ['GreyMatter', 'WholeBrain', 'SegVentricles', 'PTAU','ABETA42','TAU']:
        parents = causal_graph[t_1]
        model = MLP(input_dim=len(parents), hidden_dim=64, output_dim=1)
        model_path = os.path.join(model_dir, f'causal_mlp_{t_1}.pth')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        models[t_1] = (model, parents)
    return models


def run_inference(models, df_input, device):
    df_pred = df_input.copy()
    for t_1, (model, parents) in tqdm(models.items(), desc="Running inference"):
        X = torch.tensor(df_input[parents].values, dtype=torch.float32).to(device)
        with torch.no_grad():
            preds = model(X).cpu().numpy().flatten()
        df_pred[f"{t_1}_pred"] = preds
    return df_pred


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run inference using trained causal MLP models.")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to input CSV with baseline and target variables.")
    parser.add_argument("--causal_graph_path", type=str, required=True, help="Path to the discovered causal graph (GraphML).")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing trained MLP model weights.")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to save the output CSV with predictions.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load graph
    G = nx.read_graphml(args.causal_graph_path)
    causal_graph = convert_causal_graph(G)

    # Load input data
    df_input = pd.read_csv(args.input_csv)

    # Load trained models
    models = load_models(args.model_dir, causal_graph, device)

    # Run inference
    df_pred = run_inference(models, df_input, device)

    # Save results
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df_pred.to_csv(args.output_csv, index=False)
    print(f"Inference complete. Results saved to: {args.output_csv}")
