import torch
import torch.nn as nn

# 模拟样本的预测值和标签
batch_size = 3  # 批量大小
num_classes = 5  # 总类别数

# 预测值 (logits)，未经过 Sigmoid 激活
predictions = torch.tensor([[0.2, -1.5, 0.8, 1.2, -0.3],
                            [1.0, -0.5, 2.0, -0.8, -1.0],
                            [-1.2, 0.3, -0.5, 0.7, 1.5]], requires_grad=True)

# 标签 (multi-label ground truth)，每个类别独立是 0 或 1
labels = torch.tensor([[0, 1, 0, 1, 0],
                       [1, 0, 1, 0, 0],
                       [0, 0, 1, 1, 1]], dtype=torch.float32)

# 定义二分类交叉熵损失
# BCEWithLogitsLoss 自动将 logits 经过 Sigmoid 激活
criterion = nn.BCEWithLogitsLoss()

# 计算损失
loss = criterion(predictions, labels)

print(f"Multi-label BCE loss: {loss.item():.4f}")