# HOMGNN
This is a PyTorch implementation of the paper: [Higher-Order Masked Graph Neural Networks for Multi-Step Traffic Flow Prediction](https://www.google.com). 

## Requirements
The model is implemented using Python3 with dependencies specified in requirements.txt
## Data Preparation

### Traffic datasets
Download the METR-LA and PEMS-BAY dataset from [Google Drive](https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX) or [Baidu Yun](https://pan.baidu.com/s/14Yy9isAIZYdU__OYEQGa_g) provided by [Li et al.](https://github.com/liyaguang/DCRNN.git) . Move them into the data folder. 

```

# Create data directories
mkdir -p data/{METR-LA,PEMS-BAY}

# METR-LA
python generate_training_data.py --output_dir=data/METR-LA --traffic_df_filename=data/metr-la.h5

# PEMS-BAY
python generate_training_data.py --output_dir=data/PEMS-BAY --traffic_df_filename=data/pems-bay.h5

```

## Model Training

### Retrain Model on:
* METR-LA

```
python main.py --adj_data ./data/sensor_graph/adj_mx.pkl --data ./data/METR-LA --num_nodes 207 --order 2  --neiaccount 2

```
* PEMS-BAY

```
python main.py --adj_data ./data/sensor_graph/adj_mx_bay.pkl --data ./data/PEMS-BAY --num_nodes 325 --order 2 --neiaccount 1

```

## Citation

```
@inproceedings{yuan2022higher,
  title={Higher-Order Masked Graph Neural Networks for Multi-Step Traffic Flow Prediction},
  author={Kaixin Yuan, Jing Liu, and Jian Lou},
  booktitle={22nd IEEE International Conference on Data Mining},
  year={2022}
}
```
Our code is based on the implementation of [MTGNN](https://github.com/nnzhan/MTGNN) .
