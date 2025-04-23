# Ultra-Fast-Lane-Detection-V2
## model details
### Introducing the model
Traditional lane detection methods often rely on pixel-wise segmentation, facing challenges in efficiency and performance under difficult conditions like severe occlusions or extreme lighting. Ultra-Fast-Lane-Detection-V2 adopts a novel approach inspired by human perception, utilizing contextual and global information. It formulates lane detection as an anchor-driven ordinal classification problem using global features. This method represents lanes with sparse coordinates on hybrid anchors, significantly reducing computational cost and achieving ultra-fast speeds. Its large receptive field enables robust detection even in challenging scenarios, achieving state-of-the-art performance in both speed and accuracy.
The based model implamentation can found [here](https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2/).

Research Paper :  [Ultra Fast Deep Lane Detection with Hybrid Anchor Driven Ordinal Classification](https://arxiv.org/abs/2206.07389)
### Compare Ultra-Fast-Lane-Detection-V2 and [V1](https://github.com/cfzd/Ultra-Fast-Lane-Detection)
| Feature/Model                 | Ultra Fast Deep Lane Detection with Hybrid Anchor Driven Ordinal Classification                                    | Ultra Fast Structure-aware Deep Lane Detection                                                    |
|-------------------------------|--------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| Lane Representation           | Uses a hybrid anchor system (row and column anchors) to represent lanes with sparse coordinates.                   | Uses a row-based selection method to represent lanes as locations on predefined rows.             |
| Classification vs. Regression | Employs ordinal classification with a mathematical expectation loss to predict lane coordinates.                   | Uses classification to predict lane locations, with an expectation-based method for localization. |
| Dataset Performance           | Achieves state-of-the-art performance on four datasets (TuSimple, CULane, CurveLanes, LLAMAS).                     | Achieves state-of-the-art performance on two datasets (TuSimple, CULane).                         |
| Speed                         | Lightweight version achieves over 300 FPS.                                                                         | Achieves over 300 FPS on TuSimple dataset.                                                        |
| Main Advantages               | Hybrid anchor system reduces localization error; ordinal classification effectively handles challenging scenarios. | Row-based selection reduces computational cost; structural loss explicitly models lane structure. |
| Applicability                 | Suitable for real-time applications requiring high accuracy and robustness in challenging conditions.              | Suitable for real-time applications with a focus on speed and efficiency.                         |


---
### Basic information about the model
* Domain : Lane Detection
* Model_origin : [ufldv2_culane_res34_320x1600](https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2/)
* Binary Model : [Ultra-Fast-Lane-Detection-V2.cix]()
* input : 1x3x320x1600
* output :                
    * loc_row:    (1, 200, 72, 4)
    * exist_row:  (1, 2, 72, 4)
    * loc_col:    (1, 100, 81, 4)
    * exist_col:  (1, 2, 81, 4) 
* Params : 216.40 M
* Model Size : 826 M
---
## Performance
NPU time : 15.61 ms

CPU time : 860.4 ms
## Folder struct

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

## quantize model and export for on-device binary
```
cixbuild cfg/Ultra-Fast-Lane-Detection-V2.cfg
```
## test on onnxruntime
run onnx model on onnxruntime.
```
python3 inference_onnx_v2.py
```
## inference on NPU
run binary model on SOC.
first copy the compiled file **Ultra-Fast-Lane-Detection-V2.cix**, **test_data** & **inference_npu_v2.py** to the SOC. then run the inference_npu_v2.py script.
```
python3 inference_npu_v2.py
```
If you want to use video as input
```
python3 inference_npu_v2_video.py --video_path test_data_v2/example.mp4 --save_output 
```
## License
* the license for the original model of Ultra-Fast-Lane-Detection-V2 can be found [here](https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2)
* the license for the compiled model of Ultra-Fast-Lane-Detection-V2 can be found [cix license]