import torch
import torch.nn.functional as F

# 模拟预测值 (logits)，形状 [batch_size, num_classes]
predictions = torch.tensor([[0.2, -1.5, 0.8, 1.2, -0.3],
                            [1.0, -0.5, 2.0, -0.8, -1.0],
                            [-1.2, 0.3, -0.5, 0.7, 1.5]], requires_grad=True)

# 模拟多标签真实值 (multi-label ground truth)，形状 [batch_size, num_classes]
labels = torch.tensor([[0, 1, 0, 1, 0],
                       [1, 0, 1, 0, 0],
                       [0, 0, 1, 1, 1]])

# 逐类别计算交叉熵损失
# 转换为每个类别的二分类格式
loss = 0
for i in range(predictions.size(1)):  # 遍历每个类别
    class_pred = predictions[:, i]  # 当前类别的预测值，形状 [batch_size]
    class_label = labels[:, i]      # 当前类别的标签，形状 [batch_size]

    # 使用 cross_entropy 计算二分类交叉熵
    # 需要将标签扩展为 [0, 1] 类别
    class_label = class_label.long()  # 转为 long 类型（类别索引）
    loss += F.cross_entropy(torch.stack([1 - class_pred, class_pred], dim=1), class_label)

# 平均化损失
loss /= predictions.size(1)

print(f"Multi-label loss using cross_entropy: {loss.item():.4f}")