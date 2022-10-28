### Prepare Scannet V2 Data
1. Download Scannet data, following instruction from https://github.com/ScanNet/ScanNet

2. Prepare exported RGB images and labels. Either use scannet utils or download from https://github.com/Sekunde/3D-SIS. Save them under folder `frames_square`.

3. Run below to get the vertices, labels and bboxes as well as deeplab semantic segmentation results.
```
python batch_load_scannet_data.py
```
Training data is saved under `scannet_trian_detection_data` and semantic segmentation results are saved in `semantic_2d_results`.