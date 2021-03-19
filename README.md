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
- Weight(models) for differenct items are located at folder model, please run combin.sh to generate the weight files.

  
```Shell
# yolact_plus_base.pth will be generated under each folder
 cd model
 sh combine.sh
 ```


- Modify the model_id in eval.py (currently support 4 items: 4 5 6 7), and then run the eval.py
  
```Shell
# Quantitatively evaluate a trained model on the entire validation set. Make sure you have COCO downloaded as above.
# This should get 29.92 validation mask mAP last time I checked.
 python3 eval.py 
 ```
