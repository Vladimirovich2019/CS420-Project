# CS420-Project
Project repo for SJTU CS420 final project.

## Dataset
Stroke sequence (source): https://jbox.sjtu.edu.cn/l/G1XSvf, password: bjaf

Our dataset: https://jbox.sjtu.edu.cn/l/d1fFs6

## Environment

Install dependencies for the project:

```bash
pip install -r requirements.txt
```

Dependency for neuralline:
```
pip install ninja
```

## Train
### CNN

Modify data path, model and hyperparameters in `./configs/default.yaml`

```bash
python run_torchsketch.py
```

### RNN
```bash
python run_rnn.py
```

### Dual branch CNN-RNN with branch attention
```bash
python run_CnnRnn.py
```