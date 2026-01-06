import torch.nn as nn
import numpy as np
import torch as pt

class NNTrader(nn.Module):

    def __init__(self, n_features, verbose=False):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        self.verbose = verbose
        
    def forward(self,x):
        return self.model(x)
    
    def nn_train(self, X_train, y_train, X_val, y_val):
        model = self.model

        criterion = nn.MSELoss()          
        optimizer = pt.optim.Adam(model.parameters(), lr=1e-3)

        best_val_loss = np.inf
        patience = 20
        patience_counter = 0
        max_epochs = 500


        for epoch in range(max_epochs):
            model.train()
            optimizer.zero_grad()
            
            y_pred = model(X_train)
            train_loss = criterion(y_pred, y_train)
            
            train_loss.backward()
            optimizer.step()
            
            model.eval()
            with pt.no_grad():
                val_pred = model(X_val)
                val_loss = criterion(val_pred, y_val)
            if self.verbose:
                print(f"Epoch {epoch} | Train: {train_loss.item():.4f} | Val: {val_loss.item():.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = model.state_dict()
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                if self.verbose:
                    print("Early stopping triggered")
                break
        
        model.load_state_dict(best_state)
        self.model = model

    def nn_test(self, X_test, y_test):
        model = self.model
        model.eval()
        with pt.no_grad():
            y_pred_test = model(X_test).cpu().numpy().flatten()
            y_test = y_test.cpu().numpy().flatten()

        return (y_pred_test, y_test)

