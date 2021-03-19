# Installation
 ## Clone this repository:
```Shell
git clone https://github.com/conica-cui/yolact/.git
cd yolact
```

 ## Install dependencies:
- Pay attention to the version requirement
  
```
# Cython needs to be installed before pycocotools
pip3 install cython
pip3 install opencv-python pillow pycocotools matplotlib 
pip3 install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```

 ## Compile deformable convolutional layers
- [DCNv2] https://github.com/CharlesShang/DCNv2/tree/pytorch_1.0
- Make sure you have the latest CUDA toolkit installed from [NVidia's Website](https://developer.nvidia.com/cuda-toolkit).
  
```Shell
cd external/DCNv2
python3 setup.py build develop

# If encountered permission error, use below command instead
sudo -E python3 setup.py build develop
```


## Run and evalute different items
- Download weights and test data :
  https://pan.baidu.com/s/13egX-KK2KhLDM3UdUqUuNg  password: 1234 

- Modify the path in eval.py (eg:weight file: .pth ; png file: .png ; ply file: .ply). Make sure the png and ply file are same name.
  
```Shell
# Quantitatively evaluate a trained model on the entire validation set. Make sure you have COCO downloaded as above.
# This should get 29.92 validation mask mAP last time I checked.
 python3 eval.py 
 ```
