# Federated Zero-Shot Learning with mid-level semantic knowledge transfer
[Pattern Recognition 2024] This is the official repository for our paper: [Federated Zero-Shot Learning with mid-level semantic knowledge transfer](https://www.sciencedirect.com/science/article/pii/S0031320324005752) by Shitong Sun, Chenyang Si, Guile Wu and Shaogang Gong.

## Architecture 
![FZSL Model](./images/model.png "Model Architecture")

## Installation
 ```python
conda create -n fzsl python==3.9
 ```

## Data Preparation
   ```
    datasets/
    └── anet/
        ├── v1-2/
        │   ├── train/
        │   │   ├── xxxx.mp4
        │   │   └── xxxx.mp4
        │   └── val/
        │       ├── xxxx.mp4
        │       └── xxxx.mp4
        ├── v1-3/
        │   └── train_val/
        │       ├── xxxx.mp4
        │       └── xxxx.mp4
        └── anet_benchmark_video_id.txt
    ```
