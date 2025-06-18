import torch
from tqdm import tqdm

def MAPE(pred, true):
    mask = true.abs() > 1e-3
    return (torch.abs((pred[mask] - true[mask]) / true[mask])).mean().item()

def calculate_performance_index(
    model,
    loader,
    device,
    edge_index,
    edge_attr
):
    """
    model      : nn.Module, 예측 모델
    loader     : DataLoader, 평가할 데이터 로더
    device     : 'cpu' or 'cuda'
    edge_index : 그래프 엣지 인덱스 (torch.Tensor)
    edge_attr  : 그래프 엣지 속성 (torch.Tensor)
    
    항상 MAE, MSE, MAPE, RMSE 순으로 계산하여 출력합니다.
    """
    model.eval()

    total_mae  = 0.0
    total_mse  = 0.0
    total_mape = 0.0
    n_samples  = 0

    with torch.no_grad():
        for x_batch, y_batch in tqdm(loader, desc="Evaluating"):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # 모델 예측: [B, n_pred, E, C_out]
            pred = model(x_batch, edge_index, edge_attr)
            B = x_batch.size(0)

            # MAE
            batch_mae = torch.mean(torch.abs(pred - y_batch)).item()
            total_mae += batch_mae * B

            # MSE
            batch_mse = torch.mean((pred - y_batch) ** 2).item()
            total_mse += batch_mse * B

            # MAPE
            batch_mape = MAPE(pred, y_batch)
            total_mape += batch_mape * B

            n_samples += B

    # 평균 지표 계산
    avg_mae  = total_mae  / n_samples
    avg_mse  = total_mse  / n_samples
    avg_mape = total_mape / n_samples
    avg_rmse = (avg_mse) ** 0.5

    # 결과 출력 순서: MAE → MSE → MAPE → RMSE
    print(f"Dataset size:      {n_samples} samples")
    print(f"Average MAE:       {avg_mae:.4f}")
    print(f"Average MSE:       {avg_mse:.4f}")
    print(f"Average MAPE:      {avg_mape:.4f}")
    print(f"Average RMSE:      {avg_rmse:.4f}")
