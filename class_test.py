from tools.train import main as class_train

# ./configs/mc_config/efficientnetv2-b0_8xb32_in1k.py
# ./configs/mc_config/resnet50_8xb32_in1k.py
# ./configs/mc_config/resnet101_8xb32_in1k.py
# ./configs/mc_config/vit-base-p16_64xb64_in1k-384px.py
# ./configs/mc_config/convnext-small_32xb128_in1k.py

if __name__ == '__main__':
    class_train()