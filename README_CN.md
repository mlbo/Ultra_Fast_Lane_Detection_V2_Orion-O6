# Ultra-Fast-Lane-Detection-V2
## 模型详情
### 模型介绍
传统的车道线检测方法通常依赖于像素级分割，在严重遮挡或极端光照等复杂条件下，效率和性能面临挑战。Ultra-Fast-Lane-Detection-V2 采用了一种受人类感知启发的新方法，利用上下文和全局信息。该方法将车道线检测表述为基于锚点的序数分类问题，利用全局特征。它通过稀疏坐标在混合锚点上表示车道线，大幅降低了计算成本，实现了超快速度。其大感受野使其在复杂场景下也能实现鲁棒检测，在速度和精度上均达到业界领先水平。  
基础模型实现见 [这里](https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2/)。

论文链接：[Ultra Fast Deep Lane Detection with Hybrid Anchor Driven Ordinal Classification](https://arxiv.org/abs/2206.07389)

### Ultra-Fast-Lane-Detection-V2和[V1](https://github.com/cfzd/Ultra-Fast-Lane-Detection)对比
| 特性/模型  | Ultra-Fast-Lane-Detection-V2 | Ultra-Fast-Lane-Detection-V1    |
|--------|---------------------------------------------------------------------------------|---------------------------------------------------|
| 车道表示方式 | 使用混合锚点系统（行锚点和列锚点）表示车道，通过稀疏坐标建模车道位置                                              | 使用行选择方式，将车道表示为预定义行上的位置选择                          |
| 分类与回归  | 采用序数分类方法，利用分类的序数关系和数学期望进行车道定位                                                   | 采用分类方式，未涉及序数分类，而是通过结构损失函数优化车道的连续性和形状              |
| 数据集与性能 | 在四个数据集（TuSimple、CULane、CurveLanes、LLAMAS）上进行了测试，性能和速度均达到了 SOTA 水平               | 主要在 TuSimple 和 CULane 数据集上进行了测试，性能和速度也达到了 SOTA 水平 |
| 模型复杂度  | 通过混合锚点和序数分类进一步降低了模型复杂度，适合轻量级部署                                                  | 通过行选择和结构损失函数优化模型，复杂度相对较低，但未涉及混合锚点和序数分类            |
| 主要优点   | 混合锚点系统有效解决了单一锚点系统在不同车道类型上的定位误差问题，序数分类提升了定位精度                                    | 全局特征和结构损失函数使其在车道结构建模方面具有独特优势                      |
| 速度     | 轻量级版本速度可达 300+ FPS                                                              | 轻量级版本速度可达 322.5 FPS                               |
| 适用场景   | 适合需要高精度和实时性的车道检测任务，尤其是在复杂场景下（如严重遮挡、极端光照条件）                                      | 适合需要快速部署和实时处理的车道检测任务，尤其是在无视觉线索的场景下                |


---
### 模型基本信息
* 领域：车道线检测
* 模型来源：[ufldv2_culane_res34_320x1600](https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2/)
* 二进制模型：[Ultra-Fast-Lane-Detection-V2.cix]()
* 输入：1x3x320x1600
* 输出：                
    * loc_row:    (1, 200, 72, 4)
    * exist_row:  (1, 2, 72, 4)
    * loc_col:    (1, 100, 81, 4)
    * exist_col:  (1, 2, 81, 4) 
* 参数量：216.40 M
* 模型大小：826 M
---
## 性能
NPU 推理时间：15.61 ms

CPU 推理时间：860.4 ms

## 目录结构
```
├── cfg
│   └── Ultra-Fast-Lane-Detectionbuild_v2.cfg
├── datasets
│   └── cal_v2.npy
├── inference_npu_v2.py
├── inference_npu_v2_video.py
├── inference_onnx_v2.py
├── model
│   └── Ultra-Fast-Lane-Detection-V2.onnx
├── output_v2
│   ├── npu
│   │   ├── npu_v2_image_v2.jpg
│   │   └── npu_v2_test_v2.jpg
│   ├── npu_video
│   │   └── npu_v2_example.mp4
│   ├── onnx_v2_image_v2.jpg
│   └── onnx_v2_test_v2.jpg
├── ReadMe.md
├── test_data_v2
│   ├── example.mp4
│   ├── image_v2.jpg
│   └── test_v2.jpg
└── Ultra-Fast-Lane-Detection-V2.cix
```

## 量化模型并导出为设备端二进制
```
cixbuild cfg/Ultra-Fast-Lane-Detection-V2.cfg
```
## 在 onnxruntime 上测试
在 onnxruntime 上运行 onnx 模型。
```
python3 inference_onnx_v2.py
```
## NPU 上推理
在 SOC 上运行二进制模型。
首先将编译好的 **Ultra-Fast-Lane-Detection-V2.cix**、**test_data** 和 **inference_npu_v2.py** 拷贝到 SOC 上，然后运行 inference_npu_v2.py 脚本。

```
python3 inference_npu_v2.py
```
如需使用视频作为输入：
```
python3 inference_npu_v2_video.py --video_path test_data_v2/example.mp4 --save_output 
```
## License
* the license for the original model of Ultra-Fast-Lane-Detection-V2 can be found [here](https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2)
* the license for the compiled model of Ultra-Fast-Lane-Detection-V2 can be found [cix license]
