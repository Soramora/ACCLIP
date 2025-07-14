
import os
import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

 
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ================================================
# 1.  Encoder
# ================================================
class Encoder3D(nn.Module):
    def __init__(self, in_channels, dropout_rate=0.3):
        super().__init__()
        
        self.conv1 = nn.Conv3d(in_channels, 64, 3, 1, 1)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv3d(64, 128, 3, 1, 1)
        self.bn2 = nn.BatchNorm3d(128)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv3d(128, 256, 3, 1, 1)
        self.bn3 = nn.BatchNorm3d(256)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout3d(p=dropout_rate)

        self._initialize_weights()

        
    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        return x

    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

# ================================================
# 2.  Decoder
# ================================================
class Decoder3D(nn.Module):
    def __init__(self, out_channels, dropout_rate=0.3):
        super().__init__()
        self.conv1 = nn.Conv3d(256, 128, 3, 1, 1)
        self.bn1 = nn.BatchNorm3d(128)
        self.relu1 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.dropout1 = nn.Dropout3d(p=dropout_rate)

        self.conv2 = nn.Conv3d(128, 64, 3, 1, 1)
        self.bn2 = nn.BatchNorm3d(64)
        self.relu2 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.dropout2 = nn.Dropout3d(p=dropout_rate)

        self.conv3 = nn.Conv3d(64, out_channels, 3, 1, 1)
        self.bn3 = nn.BatchNorm3d(3)
        self.relu3 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.dropout3 = nn.Dropout3d(p=dropout_rate) 

        self._initialize_weights()
        
    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

# ================================================
# 3.  ConvGRU 
# ================================================

class ConvGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size):
        super(ConvGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        padding = kernel_size // 2

        self.reset_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.update_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.out_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        
        self.attention = nn.Conv2d(hidden_size, hidden_size, kernel_size=1)  # Capa de atención
        
        nn.init.orthogonal_(self.reset_gate.weight)
        nn.init.orthogonal_(self.update_gate.weight)
        nn.init.orthogonal_(self.out_gate.weight)
        nn.init.orthogonal_(self.attention.weight)
        nn.init.constant_(self.reset_gate.bias, 0.)
        nn.init.constant_(self.update_gate.bias, 0.)
        nn.init.constant_(self.out_gate.bias, 0.)
        nn.init.constant_(self.attention.bias, 0.)

    def forward(self, input_tensor, hidden_state):
        if hidden_state is None:
            B, C, *spatial_dim = input_tensor.size()
            hidden_state = torch.zeros([B, self.hidden_size, *spatial_dim], device=input_tensor.device)
        
        combined = torch.cat([input_tensor, hidden_state], dim=1)
        update = torch.sigmoid(self.update_gate(combined))
        reset = torch.sigmoid(self.reset_gate(combined))
        
        
        attn_weights = torch.sigmoid(self.attention(hidden_state))
        #attn_weights = torch.sigmoid(hidden_state)
        attended_hidden = hidden_state * attn_weights  # Modulación del estado oculto
        
        out = torch.tanh(self.out_gate(torch.cat([input_tensor, attended_hidden * reset], dim=1)))
        new_state = hidden_state * (1 - update) + out * update
        return new_state

class ConvGRU(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, num_layers, dropout=0.1):
        super(ConvGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.num_layers = num_layers

        cell_list = []
        for i in range(self.num_layers):
            if i == 0:
                input_dim = self.input_size
            else:
                input_dim = self.hidden_size
            cell = ConvGRUCell(input_dim, self.hidden_size, self.kernel_size)
            name = 'ConvGRUCell_' + str(i).zfill(2)

            setattr(self, name, cell)
            cell_list.append(getattr(self, name))
        
        self.cell_list = nn.ModuleList(cell_list)
        self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self, x, hidden_state=None):
        [B, seq_len, *_] = x.size()

        if hidden_state is None:
            hidden_state = [None] * self.num_layers

        current_layer_input = x 
        del x

        last_state_list = []

        for idx in range(self.num_layers):
            cell_hidden = hidden_state[idx]
            output_inner = []
            for t in range(seq_len):
                cell_hidden = self.cell_list[idx](current_layer_input[:, t, :], cell_hidden)
                cell_hidden = self.dropout_layer(cell_hidden)
                output_inner.append(cell_hidden)

            layer_output = torch.stack(output_inner, dim=1)
            current_layer_input = layer_output

            last_state_list.append(cell_hidden)

        last_state_list = torch.stack(last_state_list, dim=1)
        return layer_output, last_state_list      

# ================================================
# 4.  model
# ================================================
class ACCLIP(nn.Module):
    def __init__(self, num_predictions, scheduled_sampling_start=1.0, scheduled_sampling_end=0.2, decay_rate=0.99):
        super(ACCLIP, self).__init__()
        torch.cuda.manual_seed(233)
        self.num_predictions = num_predictions
        
        #  Parámetros para Scheduled Sampling
        self.scheduled_sampling_p = scheduled_sampling_start  # Probabilidad inicial de usar la predicción anterior
        self.scheduled_sampling_end = scheduled_sampling_end  # Probabilidad mínima de usar la predicción
        self.decay_rate = decay_rate  # Factor de decaimiento

        self.encoder = Encoder3D(in_channels=3).to(device)
        self.gru_net = ConvGRU(input_size=256, hidden_size=256, kernel_size=1, num_layers=1).to(device)
        self.decoder = Decoder3D(out_channels=3).to(device)

        self.network_pred = nn.Sequential(
                                nn.Conv2d(256, 256, kernel_size=1, padding=0),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(256, 256, kernel_size=1, padding=0)).to(device)
        
        self.relu = nn.ReLU(inplace=False)
        self._initialize_weights(self.network_pred)
    
    def forward(self, x):
        B, N, C, H, W = x.size()  # batch_size, seq_len, channels, height, width

        seq_len= N - self.num_predictions
        
        # split input and targets
        input_seq = x[:, :seq_len, :, :, :]  
        targets = x[:, seq_len:, :, :, :]
        
         # encoder
        input_encoded = self.encoder(input_seq.permute(0,2,1,3,4).contiguous())  # (B, 256, 4, H', W')
        targets_encoded = self.encoder(targets.permute(0,2,1,3,4).contiguous())
        

        # GRU
        x3 = input_encoded.view(B, input_encoded.size(-3), input_encoded.size(1), input_encoded.size(-2), input_encoded.size(-1))
    
        
        gru_output, hidden = self.gru_net(x3)
        hidden = hidden[:, -1, :]  # [B, 256, H', W']
    
        pred = []

        for i in range(targets_encoded.size(2)):  # Iterate over frames to predict
            p_tmp = self.network_pred(hidden)
            
            if self.train:
                # During training, compute MSE and apply scheduled sampling
                real_image = targets_encoded[:, :, i, :, :]
                mse_error = F.mse_loss(p_tmp, real_image)
                
                if mse_error.item() > 0.024:  # If error is high, use the real frame
                    next_input = real_image
                else:
                    # Use scheduled sampling to decide between prediction and ground truth
                    if random.random() < self.scheduled_sampling_p:
                        next_input = p_tmp
                    else:
                        next_input = real_image
            else:
                # During inference, always use the model's prediction
                next_input = p_tmp

            # Update the hidden state with the selected input
            _, hidden = self.gru_net(self.relu(next_input).unsqueeze(1), hidden.unsqueeze(0))
            hidden = hidden[:, -1, :]
            
            pred.append(p_tmp)

        # predictions to tensor
        pred = torch.stack(pred, 1)  # [B, num_predictions, 256, H', W']
        pred = pred.permute(0, 2, 1, 3, 4).contiguous()
    
        # decoder
        output = self.decoder(pred)
        output = output.permute(0, 2, 1, 3, 4).contiguous()
        
        return output
    
    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1)

    def update_scheduled_sampling(self):
        
        self.scheduled_sampling_p = max(self.scheduled_sampling_p * self.decay_rate, self.scheduled_sampling_end)
