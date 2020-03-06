import os

path = "./MobileNetV1_pretrained/"

num = 1
for file in os.listdir(path):
    os.rename(
        os.path.join(path, file), os.path.join(path, "HUB@MobileNetV1@" + file))
