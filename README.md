# Towards Single-Domain Generalized Object Detection via Causal Visual Prompts

## Main results

### Performance on SDGOD:

| Model  | Day Clear | Day Foggy | Dusk Rainy | Night Rainy | Night Clear | Avg. |                     Log                      |
|:------:|:---------:|:---------:|:----------:|:-----------:|:-----------:|:----:| :------------------------------------------: |
| Cauvis |   73.7    |   56.5    |    64.6    |    47.6     |    61.2     | 60.7 |  [log](resources/sdgod/Cauvis_DINOv2.log) |

### Comparison with SOTA PEFT Method on SDGOD:

| Model  | Backbone | Day Clear | Day Foggy | Dusk Rainy | Night Rainy | Night Clear | Avg. |                     Log                      |
|:------:|:--------:|:---------:|:---------:|:----------:|:-----------:|:-----------:|:-----------:| :------------------------------------------: |
| Cauvis | DINOv2-L |   73.7    |   56.5    |    64.6    |    47.6     |    61.2     | 60.7 |  [log](resources/sdgod/Cauvis_DINOv2.log) |
| Cauvis |  SAM-H   |   72.2    |   53.7    |    55.8    |    31.5     |    55.7     | 53.8 |  [log](resources/sdgod/Cauvis_SAM.log) |
| Cauvis | EVA02-L  |   69.7    |   50.2    |    57.6    |    34.2     |    48.1     | 52.0 |  [log](resources/sdgod/Cauvis_EVA02.log) |


### Object Detection Performance for Cityscpaes-C:

| Model  |  Detector  | Guass | Shot | Impul | Defocus | Glass | Motion | Zoom | Snow | Frost | Foggy | Bright | Contrast | Elas | Pixel | JPEGImages |   mPC   |                     Log                      |
|:------:|:----------:|:-----:|:----:|:-----:|:-------:|:-----:|:------:|:----:|:----:|:-----:|:------:|:-----:|:--------:|:-----:|:-----:|:----------:|:-------:|  :-------------: |
| Cauvis | FasterRCNN | 16.8  | 19.8 | 15.2  |  41.4   | 34.0  |  39.2  | 15.8 | 29.8 | 36.7  |  48.8  |   53.0   | 49.5 | 52.0  |    43.9    |    38.8    |    35.6    | [log](resources/cityscapes/Cauvis_cityscapes.log) |


# Validating performance on SDGOD

## Train Cauvis on 
```shell
bash tools/dist_train.sh configs/dinov2/cauvis_dinov2_dinohead_bs1x4_sdgod.py 8 --work-dir ./work_dir/cauvis --find_unused_parameters
```

## Test Cauvis on SDGOD
```shell
bash dist_test.sh configs/dinov2/cauvis_dinov2_dinohead_bs1x4_sdgod.py path/to/your.pth 8 --work-dir ./work_dir
```

# Validating Performance on Cityscapes-C
Train on Source Domain (Cityscapes)
```shell
bash tools/dist_train.sh configs/cityscapes/cauvis_bs2x4_cityscapes.py 8 --amp --work-dir ./work_dir/cauvis --find_unused_parameters
```
Test on Target Domain
```shell
bash tools/dist_test_robustness.sh configs/cityscapes/cauvis_bs2x4_cityscapes.py path/to/your/epoch_12.pth 8 --out path/to/your/xxx.pkl --work-dir ./work_dir --corruptions benchmark
```

```shell
python tools/analysis_tools/test_robustness.py configs/cityscapes/cauvis_bs2x4_cityscapes.py path/to/your.pth --out /path/to/xxx.pkl --work-dir ./work_dir --corruptions benchmark 
```
