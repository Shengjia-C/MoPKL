## ***Moving Infrared Small Target Detection***

The implementation of the paper **Motion Prior Knowledge Learning with Homogeneous Language Descriptions for Moving Infrared Small Target Detection**

## Datasets (bounding box-based)
- Datasets are available at [`ITSDT-15K`](https://drive.google.com/file/d/149HdOo8078My1FDiI8mmkH-KXJB0dvXj/view?usp=sharing) and [`IRDST`](https://pan.baidu.com/s/1ZYeJMhXMwCwj-wnjvSnHQA?pwd=cctd)(code: cctd). Or you can download `IRDST` directly from the [website](https://xzbai.buaa.edu.cn/datasets.html). 

- You need to reorganize these datasets in a format similar to the `coco_train_ITSDT.txt` and `coco_val_ITSDT.txt` files we provided (`.txt files` are used in training).  We provide the `.txt files` for ITSDT-15K and IRDST.
For example:
```python
train_annotation_path = '/home/ITSDT-15K/coco_train_ITSDT.txt'
val_annotation_path = '/home/ITSDT-15K/coco_val_ITSDT.txt'
```
- Or you can generate a new `txt file` based on the path of your datasets. `.txt files` (e.g., `coco_train_ITSDT.txt`) can be generated from `.json files` (e.g., `instances_train2017.json`). We also provide all `.json files` for [`ITSDT-15K`](https://drive.google.com/file/d/149HdOo8078My1FDiI8mmkH-KXJB0dvXj/view?usp=sharing) and [`IRDST`](https://pan.baidu.com/s/1ZYeJMhXMwCwj-wnjvSnHQA?pwd=cctd)(code: cctd).

``` python 
python utils_coco/coco_to_txt.py
```

- The folder structure should look like this:
```
ITSDT-15K
├─instances_train2017.json
├─instances_test2017.json
├─coco_train_ITSDT.txt
├─coco_val_ITSDT.txt
├─images
│   ├─1
│   │   ├─0.bmp
│   │   ├─1.bmp
│   │   ├─2.bmp
│   │   ├─ ...
│   ├─2
│   │   ├─0.bmp
│   │   ├─1.bmp
│   │   ├─2.bmp
│   │   ├─ ...
│   ├─3
│   │   ├─ ...
```


## Prerequisite

* python==3.11.8
* pytorch==2.1.1
* torchvision==0.16.1
* numpy==1.26.4
* opencv-python==4.9.0.80
* scipy==1.13
* Tested on Ubuntu 20.04, with CUDA 11.8, and 1x NVIDIA 3090.


## Usage of MoPKL

### Language Descriptions

- We provide encoded [language description embedding representations](https://pan.baidu.com/s/1GOxLlOiXHsRuHUwYr3Fh5g?pwd=xbet)(code: xbet) of `ITSDT-15K` and `IRDST` datasets. 
There are two embedded representations in this file: `emb_train_IRDST.pkl` and `emb_train_IRDST.pkl`.

- We also provide initial language description [text files](https://pan.baidu.com/s/17OOSx0Kfoc5N-aeQU6VcAw?pwd=bn38)(code: bn38) that you can explore further with vision-language models.
- Take the ITSDT-15K dataset as an example, modify the path of the `dataloader_for_ITSDT` for language description embedding representations:
```python
#Path to your emb_train_ITSDT.pkl

description = pickle.load(open('/home/MoPKL/emb_train_ITSDT.pkl', 'rb'))
```

### Train
- Note: Please use different `dataloader` for different datasets. For example, to train the model on ITSDT dataset, enter the following command: 
```python
CUDA_VISIBLE_DEVICES=0 python train_ITSDT.py 
```

### Test
- Usually `model_best.pth` is not necessarily the best model. The best model may have a lower val_loss or a higher AP50 during verification.
```python
"model_path": '/home/MoPKL/logs/model.pth'
```
- You need to change the path of the `json file` of test sets. For example:
```python
#Use ITSDT-15K dataset for test

cocoGt_path         = '/home/public/ITSDT-15K/instances_test2017.json'
dataset_img_path    = '/home/public/ITSDT-15K/'
```
```python
python test.py
```

### Visulization
- We support `video` and `single-frame image` prediction.
```python
# mode = "video" #Predict a sequence

mode = "predict"  #Predict a single-frame image 
```
```python
python predict.py
```

## Results
- For bounding box detection, we use COCO's evaluation metrics:

<table>
  <tr>
    <th>Method</th>
    <th>Dataset</th>
    <th>mAP50 (%)</th>
    <th>Precision (%)</th>
    <th>Recall (%)</th>
    <th>F1 (%)</th>
    <th>Download</th>
  </tr>
  <tr>
    <td align="center">MoPKL</td>
    <td align="center">ITSDT-15K</td>
    <td align="center">79.78</td>
    <td align="center">93.29</td>
    <td align="center">86.80</td>
    <td align="center">89.92</td>
    <td rowspan="3" align="center">
      <a href="https://pan.baidu.com/s/1gmvsyKZsqir70UpEnjL3Nw?pwd=pchd">Baidu</a> (code: pchd)
      <br>
    </td>
  </tr>
  <tr>
    <td align="center">MoPKL</td>
    <td align="center">IRDST</td>
    <td align="center">74.54</td>
    <td align="center">89.04</td>
    <td align="center">84.74</td>
    <td align="center">86.84</td>
  </tr>
 </table>



- PR curves on ITSDT-15K and IRDST datasets in this paper.
-We provide the [results](https://pan.baidu.com/s/1aQoImRzJOAuhNnoaQMKEXw?pwd=4ves) (4ves)  on 'ITSDT-15K' and 'IRDST', and you can plot them using Python and matplotlib.

<img src="/README/PR.png" width="700px">


## Contact
If any questions, kindly contact with Shengjia Chen via e-mail: csj_uestc@126.com.

## References
1. S. Chen, L. Ji, J. Zhu, M. Ye and X. Yao, "SSTNet: Sliced Spatio-Temporal Network With Cross-Slice ConvLSTM for Moving Infrared Dim-Small Target Detection," in IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1-12, 2024, Art no. 5000912, doi: 10.1109/TGRS.2024.3350024. 
2. Bingwei Hui, Zhiyong Song, Hongqi Fan, et al. A dataset for infrared image dim-small aircraft target detection and tracking under ground / air background[DS/OL]. V1. Science Data Bank, 2019[2024-12-10]. https://cstr.cn/31253.11.sciencedb.902. CSTR:31253.11.sciencedb.902.
3. Ruigang Fu, Hongqi Fan, Yongfeng Zhu, et al. A dataset for infrared time-sensitive target detection and tracking for air-ground application[DS/OL]. V2. Science Data Bank, 2022[2024-12-10]. https://cstr.cn/31253.11.sciencedb.j00001.00331. CSTR:31253.11.sciencedb.j00001.00331.


## Citation

If you find this repo useful, please cite our paper. 

```
```