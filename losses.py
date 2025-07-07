# Aquí se definirá combined_loss y OpticalFlowLoss
# ======================================================
# 5. Función de pérdida suma ponderada de MSE y SSIM
# ======================================================

def combined_loss(img1, img2, alpha):
    mse = F.mse_loss(img1, img2)
    ssim = 1 - structural_similarity_index_measure(img1, img2)
    return alpha * mse + (1 - alpha) * ssim


    #########################################33

class OpticalFlowLoss(torch.nn.Module):
    def __init__(self):
        super(OpticalFlowLoss, self).__init__()

    def compute_optical_flow(self, img1, img2):
        """Calcula flujo óptico denso entre dos imágenes en escala de grises."""
        img1_np = img1.squeeze().cpu().detach().numpy()
        img2_np = img2.squeeze().cpu().detach().numpy()
        
        flow = cv2.calcOpticalFlowFarneback(img1_np, img2_np, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        return torch.tensor(flow, dtype=torch.float32, device=img1.device)

    def forward(self, real_frames, pred_frames):
        """Calcula la pérdida basada en flujo óptico entre frames reales y predichos."""
        batch_size, num_frames, _, _, _ = real_frames.shape
        loss = 0.0

        for i in range(num_frames - 1):  # Iterar sobre pares de frames consecutivos
            img1_real = torch.mean(real_frames[:, i], dim=1)  # (batch, H, W)
            img2_real = torch.mean(real_frames[:, i+1], dim=1)

            img1_pred = torch.mean(pred_frames[:, i], dim=1)
            img2_pred = torch.mean(pred_frames[:, i+1], dim=1)

            # Calcular flujos ópticos
            flow_real = self.compute_optical_flow(img1_real, img2_real)
            flow_pred = self.compute_optical_flow(img1_pred, img2_pred)

            # Diferencia de flujo
            flow_loss = F.l1_loss(flow_real, flow_pred)
            loss += flow_loss

        return loss / (num_frames - 1)


def loss_function(pred, target, alpha=0.7, beta=0.2, weights=None):
    """
    Calcula una pérdida combinada entre MSE, SSIM y Flujo Óptico.

    Args:
        pred (Tensor): Predicción del modelo. Dim: (batch, num_predictions, C, H, W)
        target (Tensor): Imagen real esperada. Dim: (batch, num_predictions, C, H, W)
        alpha (float): Peso de la pérdida MSE en la combinación.
        beta (float): Peso de la pérdida basada en flujo óptico.
        weights (Tensor, opcional): Pesos para cada imagen en la secuencia.

    Returns:
        loss (Tensor): Pérdida total combinada.
    """
    # 🔹 Calculamos la pérdida MSE por imagen
    mse_losses = F.mse_loss(pred, target, reduction='none').mean(dim=(2,3,4))  # (batch, num_predictions)
    
    # 🔹 Calculamos el SSIM por imagen
    ssim_losses = torch.zeros_like(mse_losses)   # (batch, num_predictions)
    for i in range(pred.shape[1]):  # Iterar sobre num_predictions
        for j in range(pred.shape[0]):  # Iterar sobre batch
            ssim_losses[j, i] = torch.tensor(
                ssim(
                    pred[j, i].detach().cpu().numpy().transpose(1, 2, 0),
                    target[j, i].detach().cpu().numpy().transpose(1, 2, 0),
                    channel_axis=2,
                    data_range=1.0
                ),
                device=pred.device, dtype=torch.float32  # 🔹 Convertir a tensor en la GPU
            )
    ssim_losses = 1 - ssim_losses  # Convertir a una "pérdida" (1 - SSIM)

    # 🔹 Aplicar pesos (si se proporciona)
    if weights is not None:
        weights = weights.view(1, -1, 1, 1, 1)  # Expandir dimensiones
        mse_losses = mse_losses * weights  # Aplicar pesos a MSE
        ssim_losses = ssim_losses * weights.squeeze()  # Aplicar pesos a SSIM

    # 🔹 Calculamos la pérdida de flujo óptico
    optical_flow_loss = OpticalFlowLoss().to(pred.device)
    flow_loss_value = optical_flow_loss(target, pred)

    # 🔹 Combinar MSE, SSIM y Optical Flow Loss
    mse_loss_mean = mse_losses.mean()  # Promediar MSE
    ssim_loss_mean = ssim_losses.mean()  # Promediar SSIM

    loss = alpha * mse_loss_mean + (1 - alpha) * ssim_loss_mean + beta * flow_loss_value
    return loss