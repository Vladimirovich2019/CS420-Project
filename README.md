# CS420-Project
Project repo for SJTU CS420 final project.

## Dataset
Stroke sequence (source): https://jbox.sjtu.edu.cn/l/G1XSvf, password: bjaf

PNG images: https://jbox.sjtu.edu.cn/l/O1qaEt

Pickled dataset (image + stroke + label): https://jbox.sjtu.edu.cn/l/h1PZLW

## Environment

Install full dependencies for the project:

```bash
pip install -r requirements.txt
```

Install the transformers package:
```
pip install -r requirements_transformers.txt
pip install git+https://github.com/huggingface/transformers
```
You can run the following command to test if the package has been installed successfully:
```python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('I love you'))"```

Install dependencies for image classification:
```
pip install -r requirements_image_classification.txt
```

## Train
### CNN
Suggested models: microsoft/resnet-18, microsoft/resnet-34, microsoft/resnet-50, microsoft/resnet-101, microsoft/resnet-152

```
python run_resnet.py \
--train_dir dataset_images/train \
--dev_dir dataset_images/dev \
--test_dir dataset_images/test \
--model_name_or_path microsoft/resnet-50 \
--per_device_train_batch_size 128 \
--per_device_eval_batch_size 128 \
--learning_rate 5e-5 \
--num_train_epochs 1 \
--output_dir results \
--cache_dir cache
```
