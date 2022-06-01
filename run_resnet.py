
__reference__ = '''
@inproceedings{wolf-etal-2020-transformers,
    title = "Transformers: State-of-the-Art Natural Language Processing",
    author = "Thomas Wolf and Lysandre Debut and Victor Sanh and Julien Chaumond and Clement Delangue and Anthony Moi and Pierric Cistac and Tim Rault and Rémi Louf and Morgan Funtowicz and Joe Davison and Sam Shleifer and Patrick von Platen and Clara Ma and Yacine Jernite and Julien Plu and Canwen Xu and Teven Le Scao and Sylvain Gugger and Mariama Drame and Quentin Lhoest and Alexander M. Rush",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
    month = oct,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-demos.6",
    pages = "38--45"
}

https://github.com/huggingface/transformers
'''

import argparse
import json
import math
import pickle
import os

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
from tqdm.auto import tqdm

import transformers
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    ConvNextFeatureExtractor,
    AutoModelForImageClassification,
    SchedulerType,
    get_scheduler,
)
from metrics import Accuracy


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a Transformers model on an image classification dataset")

    parser.add_argument("--train_dir", type=str, default=None, help="A folder containing the training data.")
    parser.add_argument("--dev_dir", type=str, default=None, help="A folder containing the validation data.")
    parser.add_argument("--test_dir", type=str, default=None, help="A folder containing the test data.")

    parser.add_argument(
        "--model_name_or_path", type=str, default="google/vit-base-patch16-224-in21k",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--per_device_train_batch_size", type=int, default=16, help="Batch size for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size", type=int, default=16, help="Batch size for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--eval_steps", type=int, default=100, help="Number of update steps between two evaluations")
    parser.add_argument(
        "--lr_scheduler_type", type=SchedulerType, default="linear", help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")

    args = parser.parse_args()

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    print('arguments parsed')
    return args


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = parse_args()

    # If passed along, set the training seed now.
    if args.seed is not None:
        torch.manual_seed(args.seed)

    # Handle the repository creation
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # Get the dataset
    with open('dataset.pickle', 'rb') as f:
        dataset = pickle.load(f)
    print(dataset)

    n_train = dataset["train"].num_rows
    n_dev = dataset["dev"].num_rows
    n_test = dataset["test"].num_rows
    print(f'training samples: {n_train}\nvalidation samples: {n_dev}\ntest samples: {n_test}')
    print(dataset["train"].features)

    # Load pretrained model and feature extractor
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=25,
        finetuning_task="image-classification",
    )
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_name_or_path)
    model = AutoModelForImageClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        ignore_mismatched_sizes=True,
    ).to(device)
    print('model loaded')
    print(model)

    # Define torchvision transforms to be applied to each image.
    normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
    train_transforms = Compose(
        [
            RandomResizedCrop(feature_extractor.size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )
    val_transforms = Compose(
        [
            Resize(feature_extractor.size),
            CenterCrop(feature_extractor.size),
            ToTensor(),
            normalize,
        ]
    )

    def preprocess_train(example_batch):
        """Apply _train_transforms across a batch."""
        example_batch["pixel_values"] = [train_transforms(image.convert("RGB")) for image in example_batch["image"]]
        return example_batch

    def preprocess_val(example_batch):
        """Apply _val_transforms across a batch."""
        example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch["image"]]
        return example_batch

    # Set the training transforms
    train_dataset = dataset["train"].with_transform(preprocess_train)
    # Set the validation and test transforms
    eval_dataset = dataset["dev"].with_transform(preprocess_val)
    test_dataset = dataset["test"].with_transform(preprocess_val)

    # DataLoaders creation:
    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples]).to(device)
        labels = torch.tensor([example["labels"] for example in examples]).to(device)
        return {"pixel_values": pixel_values, "labels": labels}

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=collate_fn, batch_size=args.per_device_eval_batch_size)
    test_dataloader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=args.per_device_eval_batch_size)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=max_train_steps,
    )

    # Get the metric function
    metric = Accuracy()

    # Train!
    total_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps

    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num Epochs = {args.num_train_epochs}")
    print(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    print(f"  Total optimization steps = {max_train_steps}")
    log = open('logs/train_resnet.txt', 'w')
    log_dev = open('logs/validation_resnet.txt', 'w')
    log_test = open('logs/test_resnet.txt', 'w')

    progress_bar = tqdm(range(max_train_steps))
    completed_steps = 0

    for epoch in range(args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            # We keep track of the loss at each epoch
            loss = loss / args.gradient_accumulation_steps
            print(f'Epoch: {epoch}, step: {step}, loss: {float(loss)}', file=log)
            loss.backward()
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if step % args.eval_steps == 0 and step > 0 or step == len(train_dataloader) - 1:
                print('Running evaluation')
                dev_bar = tqdm(range(len(eval_dataloader)))
                model.eval()
                for step_dev, batch_dev in enumerate(eval_dataloader):
                    with torch.no_grad():
                        outputs = model(**batch_dev)
                    predictions = outputs.logits.argmax(dim=-1)
                    predictions, references = (predictions, batch_dev["labels"])
                    metric.add_batch(
                        predictions=predictions,
                        references=references,
                    )
                    dev_bar.update(1)

                eval_metric = metric.compute()
                print(f"Epoch: {epoch}, step: {step}: {eval_metric}")
                print(f"Epoch: {epoch}, step: {step}: {eval_metric}", file=log_dev)
                model.train()

            if completed_steps >= max_train_steps:
                break

        output_dir = f"epoch_{epoch}"
        if args.output_dir is not None:
            output_dir = os.path.join(args.output_dir, output_dir)
            model.save_pretrained(output_dir)

    print('Running test')
    test_bar = tqdm(range(len(test_dataloader)))
    model.eval()
    for step_test, batch_test in enumerate(test_dataloader):
        with torch.no_grad():
            outputs = model(**batch_test)
        predictions = outputs.logits.argmax(dim=-1)
        predictions, references = (predictions, batch_test["labels"])
        metric.add_batch(
            predictions=predictions,
            references=references,
        )
        test_bar.update(1)

    eval_metric = metric.compute()
    print(f"Test accuracy: {eval_metric}")
    print(f"Test accuracy: {eval_metric}", file=log_test)

    if args.output_dir is not None:
        model.save_pretrained(args.output_dir)
        feature_extractor.save_pretrained(args.output_dir)

    if args.output_dir is not None:
        with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
            json.dump({"eval_accuracy": eval_metric["accuracy"]}, f)

    log.close()
    log_dev.close()
    log_test.close()


if __name__ == "__main__":
    main()
