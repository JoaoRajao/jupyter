import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
import pygad

# --- Define as células xLSTM e ForecastModel ---

class sLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.z_gate = nn.Linear(input_size, hidden_size)
        self.i_gate = nn.Linear(input_size, hidden_size)
        self.f_gate = nn.Linear(input_size, hidden_size)
        self.o_gate = nn.Linear(input_size, hidden_size)
        self.zr_gate = nn.Linear(hidden_size, hidden_size, bias=False)
        self.ir_gate = nn.Linear(hidden_size, hidden_size, bias=False)
        self.fr_gate = nn.Linear(hidden_size, hidden_size, bias=False)
        self.or_gate = nn.Linear(hidden_size, hidden_size, bias=False)
        self.register_buffer('ht_1', torch.zeros(1, hidden_size))
        self.register_buffer('ct_1', torch.zeros(1, hidden_size))
        self.register_buffer('nt_1', torch.zeros(1, hidden_size))

    def init_states(self, batch_size):
        device = self.ht_1.device
        self.ht_1 = torch.zeros(batch_size, self.ht_1.size(1), device=device)
        self.ct_1 = torch.zeros(batch_size, self.ct_1.size(1), device=device)
        self.nt_1 = torch.zeros(batch_size, self.nt_1.size(1), device=device)

    def forward(self, x):
        x = self.dropout(x)
        z = torch.tanh(self.z_gate(x) + self.zr_gate(self.ht_1))
        i = torch.exp(self.i_gate(x) + self.ir_gate(self.ht_1))
        f = torch.exp(self.f_gate(x) + self.fr_gate(self.ht_1))
        o = torch.sigmoid(self.o_gate(x) + self.or_gate(self.ht_1))
        c = f * self.ct_1 + i * z
        n = f * self.nt_1 + i
        h = o * (c / n)
        self.ct_1 = c.detach()
        self.nt_1 = n.detach()
        self.ht_1 = h.detach()
        return h

class mLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.i_gate = nn.Linear(input_size, hidden_size)
        self.f_gate = nn.Linear(input_size, hidden_size)
        self.o_gate = nn.Linear(input_size, hidden_size)
        self.query  = nn.Linear(input_size, hidden_size)
        self.key    = nn.Linear(input_size, hidden_size)
        self.value  = nn.Linear(input_size, hidden_size)
        self.register_buffer('Ct_1', torch.zeros(1, hidden_size))
        self.register_buffer('nt_1', torch.zeros(1, hidden_size))

    def init_states(self, batch_size):
        device = self.Ct_1.device
        self.Ct_1 = torch.zeros(batch_size, self.Ct_1.size(1), device=device)
        self.nt_1 = torch.zeros(batch_size, self.nt_1.size(1), device=device)

    def forward(self, x):
        x = self.dropout(x)
        i = torch.exp(self.i_gate(x))
        f = torch.exp(self.f_gate(x))
        o = torch.sigmoid(self.o_gate(x))
        q = self.query(x)
        k = self.key(x) / (self.key.weight.shape[1] ** 0.5)
        v = self.value(x)
        C = f * self.Ct_1 + i * v * k
        n = f * self.nt_1 + i * k
        h = o * (C * q / torch.max(torch.abs(n * q)))
        self.Ct_1 = C.detach()
        self.nt_1 = n.detach()
        return h

class xLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.mlstm = mLSTMCell(input_size, hidden_size, dropout)
        self.slstm = sLSTMCell(input_size, hidden_size, dropout)

    def init_states(self, batch_size, seq_len):
        self.mlstm.init_states(batch_size)
        self.slstm.init_states(batch_size)

    def forward(self, x):
        outputs = []
        B, T, _ = x.size()
        self.init_states(B, T)
        for t in range(T):
            xt = x[:, t, :]
            hm = self.mlstm(xt)
            hs = self.slstm(xt)
            outputs.append((hm + hs).unsqueeze(1))
        return torch.cat(outputs, dim=1)

class ForecastModel(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.0):
        super().__init__()
        self.xlstm = xLSTM(input_size, hidden_size, dropout)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = self.xlstm(x)
        out = out[:, -1, :]
        return self.linear(out).squeeze(-1)

def load_data(path, seq_length=12, test_ratio=0.2):
    df = pd.read_csv(path, parse_dates=['REPORT_DATE'])
    ts = df.set_index('REPORT_DATE').resample('M').size().to_frame('count')
    scaler = MinMaxScaler()
    ts['scaled'] = scaler.fit_transform(ts[['count']])
    values = ts['scaled'].values
    X, y = [], []
    for i in range(len(values) - seq_length):
        X.append(values[i:i+seq_length])
        y.append(values[i+seq_length])
    X = np.array(X)[..., None]
    y = np.array(y)
    split = int((1 - test_ratio) * len(X))
    return (torch.tensor(X[:split], dtype=torch.float32),
            torch.tensor(y[:split], dtype=torch.float32),
            torch.tensor(X[split:], dtype=torch.float32),
            torch.tensor(y[split:], dtype=torch.float32))

def fitness_func(ga, sol, sol_idx):
    hidden_size = int(sol[0])
    lr = sol[1]
    batch_size = int(sol[2])
    dropout = sol[3]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_train, y_train, X_val, y_val = load_data(
        r'C:\Users\peder\Downloads\major-crime-indicators.csv'
    )
    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val),
                            batch_size=batch_size)
    model = ForecastModel(1, hidden_size, dropout).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    for _ in range(5):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = criterion(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    losses = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            losses.append(criterion(model(xb), yb).item())
    val_loss = np.mean(losses)
    return 1.0 / (val_loss + 1e-8)

if __name__ == "__main__":
    ga = pygad.GA(
        num_generations=10,
        num_parents_mating=3,
        fitness_func=fitness_func,
        sol_per_pop=8,
        num_genes=4,
        gene_space=[
            list(range(32, 129, 32)),
            {'low': 1e-4, 'high': 1e-2},
            list(range(8, 65, 8)),
            {'low': 0.0, 'high': 0.5}
        ],
        mutation_percent_genes=20
    )
    ga.run()
    sol, fit, _ = ga.best_solution()
    print(f"Melhor solução -> hidden_size={int(sol[0])}, lr={sol[1]:.6f}, batch_size={int(sol[2])}, dropout={sol[3]:.2f}")
