# Raw Python Kohonen Self Organizing Maps (SOM)

This implementation was created for UTFPR Applied Intelligent Systems classes, as part of
my CS bachelor's.

The data used for POC was obtained from the UCI public ML repository: 
https://archive.ics.uci.edu/ml/datasets.php, with corresponding names: 
* Breast cancer
* Lung cancer
* Wine

> Script usage:
```
python main.py --data_dir [relative_path] --map_height [map_height] --map_width [map_width] \
--dist [0/1/2]
```
> Examples:
```
python main.py --map_width 3 --map_height 1
```

```
python main.py --data_dir ../data/lung-cancer/lung-cancer.data \
--output_dir ../data/outputs/lung_cancer_outputs.csv --header False --map_width 3 --map_height 1
```
