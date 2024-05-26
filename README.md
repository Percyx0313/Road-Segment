# Road-Segment
Jetbot's Road segmentation model

# Dataset
We use data set from kaggle
```
https://www.kaggle.com/datasets/sovitrath/road-lane-segmentation-train-test-split
```
## LabelMe dataset

### Directory structure
```
data
|---data_annotated
|   |---<image 0>.jpg
|   |---<image 0>.json
|   |---...
|---data_dataset_voc
|   |---JPEGImages
|   |---SegmentationClass
|   |---SegmentationClassNpy
|   |---SegmentationClassVisualization
|   |---class_names.txt
|---labels.txt
```
### Label all images in data_annotated directory
```
cd data
labelme data_annotated --labels labels.txt --nodata --validatelabel exact
```

### Create data_dataset_voc directory
```
cd data
python labelme2voc.py data_annotated data_dataset_voc --labels labels.txt --noobject
```
