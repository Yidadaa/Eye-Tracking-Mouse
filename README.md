# Eye Tracking Mouse

## dependency
```
pip install tensorflow
pip install mtcnn opencv-python pyWinhook pyWin32
```

## TODO List
- [x] 设计模型以及数据收集脚本
- [x] 收集数据，查看模型效果
- [ ] 加入人脸特征
- [ ] 只追踪登记的人脸

## Experiement Log
### Version 1, Train as Classifier
use eye feature, face feature and grid feature, with pretrained mobilenet v3 backbones.

Dataset:
1197 images.

Result:
epoch    train_loss    valid_acc    valid_loss     dur
  100        0.0000       0.5000        4.0734  6.0859

### Version 1.2, Train as Classifier
- [v]eye feature
- [x]face feature
- [v]grid feature
- [v]pretrained mobilenet v3 backbones

Dataset:
1197 images.