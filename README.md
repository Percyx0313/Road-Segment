# Road-Segment
Jetbot's Road segmentation model

# Dataset
## Roadlane Dataset
```
https://www.kaggle.com/datasets/sovitrath/road-lane-segmentation-train-test-split
```

### Directory structure
```
data
|---roadlane
    |---train
    |   |---images
    |   |---masks
    |---val
        |---images
        |---masks
```

## LabelMe dataset

### Directory structure
```
data
|---labelme
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
cd data/labelme
labelme data_annotated --labels labels.txt --nodata --validatelabel exact
```

### Create data_dataset_voc directory
```
cd data/labelme
python labelme2voc.py data_annotated data_dataset_voc --labels labels.txt --noobject
```
