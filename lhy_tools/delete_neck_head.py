import torch

def load_and_filter_backbone_weights(checkpoint_path, output_path):
    """
    加载预训练权重，过滤并重命名 backbone 部分的权重，然后保存为新的文件。

    Args:
        checkpoint_path (str): 预训练权重文件的路径。
        output_path (str): 保存处理后权重的路径。
    """
    # 加载预训练权重
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # 过滤并重命名权重
    new_state_dict = {}
    for key, value in checkpoint.items():
        if key.startswith('backbone.'):
            # 去掉 'backbone.' 前缀
            new_key = key.replace('backbone.', '')
            new_state_dict[new_key] = value

    # 保存处理后的权重
    torch.save(new_state_dict, output_path)
    print(f"Backbone weights have been saved to {output_path}")

# 使用示例
checkpoint_path = '/mnt/mydisk/medical_seg/fwwb_a007/checkpoints/convnext-small_32xb128_in1k_20221207-4ab7052c.pth'  # 替换为你的权重文件路径
output_path = '/mnt/mydisk/medical_seg/fwwb_a007/checkpoints/convnext-small_32xb128_in1k.pth'  # 替换为你想保存的路径
load_and_filter_backbone_weights(checkpoint_path, output_path)