# GraphNet

This project is a code implementation of GraphNet, focusing on the task of cross-temporal reconstruction and prediction of graph structure data. Based on an innovative cross-temporal graph neural network architecture, it solves core problems such as missing data and modeling of spatiotemporal correlations.
Due to the large size of the data file, you will need to perform the decompression process independently. Meanwhile, _1.z01, _1.z02, and _1.zip are split archive files.

## Requirement

```
conda create -n graphnet python=3.9
conda activate graphnet
conda install -c conda-forge numpy pandas scikit-learn matplotlib networkx pickle5
pip install torch==1.12.1
```

## Usage

- Data preprocessing

```
python load_data.py
```

- Basic model training

```
python simple_train.py
```

- Model training & testing

```
python main.py
```

- Results visualization

```
python plot.py
```

## Experimental settings

- Optimizer: Adam
- Learning Rate: 0.001
- Weight Decay: 0.0001
- Batch Size: 512
- Seed: 42
- Number of training rounds: 250
- Early stop patience value: 10