# Eye Tracking Mouse

## dependency
```
pip install tensorflow
pip install mtcnn opencv-python pyWinhook pyWin32
```

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

Result:
epoch    train_loss    valid_acc    valid_loss     dur
  100        0.0001       0.4693        3.8692  2.2839

### Version 1.3, Train as Classifier
- [v]eye feature
- [x]face feature
- [v]grid feature
- [x]pretrained mobilenet v3 backbones

Dataset:
1197 images.

Result:
epoch    train_loss    valid_acc    valid_loss     dur
  100        0.0001       0.4693        3.8692  2.2839

### Version 1.3, Train as Tree Classifier
- [v] face / nose / mouth position features
- lightgbm

Dataset:
1197 images.

Result:
acc: 0.2791666666666667
