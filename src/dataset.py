from pathlib import Path
from torch.utils.data import Dataset
from transformers import (
    PreTrainedTokenizer,
    AutoTokenizer
)
import pandas as pd
import os
import torch
import logging
import random
import random
from sklearn.model_selection import train_test_split
import pickle
from datasets import load_dataset as _load_dataset
from augment import Augment
from datasets import Dataset, DatasetDict


def load_dataset(data_path, tokenizer: PreTrainedTokenizer, cache_dir=None, args=None):

    assert os.path.isfile(data_path), data_path
    logging.info("[CL] Creating features from dataset file at %s", data_path)

    data_path = {
        "train": data_path.as_posix()
    }
    dataset = _load_dataset("csv", data_files=data_path, cache_dir=cache_dir)

    aug = Augment(args, p=args.augment_prob)

    # if os.path.isfile(cache_dir / 'synonym_dict.json'):
    #     logging.info(f"loading synonym dict for augmentation: {str(cache_dir / 'synonym_dict.json')}")
    #     aug.load_synonym_dict(cache_dir / 'synonym_dict.json')

    def _transform(example):
        example['augment'] = aug.apply_augment(example["question"], args.augmentation_strategy)
        if args.lower:
            example['augment'] = example['augment'].lower()
            example['question'] = example['question'].lower()
        return example

    # dataset.set_transform(_transform)
    dataset = dataset.map(
        _transform,
        num_proc=8
    )

    def _tokenize(examples):
        ret = {}
        for k, v in tokenizer(examples["question"], truncation=True).items():
            ret[k] = v
        for k, v in tokenizer(examples["augment"], truncation=True).items():
            ret['aug-'+k] = v
        return ret

    remove_columns = ["question", "level_1_knowledge", "level_2_knowledge", "augment"]
    system2 = False

    if "level_4_knowledge" in dataset.column_names["train"]:
        remove_columns.append("level_3_knowledge")
        logging.info("[Data] use system2 dataset")
        system2 = True
    dataset = dataset.map(
        _tokenize,
        batched=True,
        remove_columns=remove_columns,
        num_proc=8
    )
    if system2:
        dataset = dataset.rename_column("level_4_knowledge", "level_3_knowledge")
    # logging.info(f"save synonym dict for augmentation: {str(cache_dir / 'synonym_dict.json')}")
    # aug.save_synonym_dict(cache_dir / 'synonym_dict.json')
    return dataset.with_format("torch")


def load_similarity_dataset(similarity_path, question_path, tokenizer, args):
    sim_df = pd.read_csv(similarity_path)
    ques_df = pd.read_csv(question_path)
    id_filter = set(sim_df['id1'].unique()) | set(sim_df['id2'].unique())
    ques_df = ques_df[ques_df['id'].isin(id_filter)]

    id_dict = {id: i for i, id in enumerate(ques_df['id'].tolist())}

    if args.lower:
        ques_df['question'] = ques_df['question'].apply(lambda x : x.lower())

    dataset = Dataset.from_pandas(ques_df)
    remove_columns = dataset.column_names

    def _tokenize(examples):
        return tokenizer(examples["question"], truncation=True)
    dataset = dataset.map(
        _tokenize,
        batched=True,
        remove_columns=remove_columns,
        num_proc=8
    )
    return dataset.with_format("torch"), id_dict


def load_eval_dataset(train_df, test_df, tokenizer, args):
    if args.lower:
        train_df['question'] = train_df['question'].apply(lambda x : x.lower())
        test_df['question'] = test_df['question'].apply(lambda x : x.lower())

    dataset = DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "test": Dataset.from_pandas(test_df)
    })
    remove_columns = dataset["train"].column_names
    remove_columns.remove("label")

    def _tokenize(examples):
        return tokenizer(examples["question"], truncation=True)
    dataset = dataset.map(
        _tokenize,
        batched=True,
        remove_columns=remove_columns,
        num_proc=8
    )

    return dataset.with_format("torch")


def load_test_dataset(data_path, tokenizer: PreTrainedTokenizer, args):
    if args.test_task == 'similarity':
        sim_data_path = data_path / 'similarity.csv'
        data_path = data_path / 'question.csv'
        return load_similarity_dataset(sim_data_path, data_path, tokenizer, args)
    elif args.test_task == 'concepts':
        data_path = data_path / 'task.csv'
        df = pd.read_csv(data_path).dropna(subset=['question', f"level_{args.test_know_level}_knowledge"]).reset_index()
        n_classes = df[f"level_{args.test_know_level}_knowledge"].max() + 1
        logging.info(f"[test-con] Num. of level{args.test_know_level} knowledge concepts: {n_classes}")
        df.rename({ f"level_{args.test_know_level}_knowledge": "label"}, axis=1, inplace=True)
        train_idx, test_idx = train_test_split(df.index,test_size=0.2, random_state=0)
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        return load_eval_dataset(train_df, test_df, tokenizer, args), n_classes
    elif args.test_task == 'difficulty':
        data_path = data_path / 'difficulty.csv'
        assert os.path.isfile(data_path), data_path
        df = pd.read_csv(data_path).dropna(subset=['question', "difficulty"]).reset_index()
        df.rename({ "difficulty": "label"}, axis=1, inplace=True)
        train_idx, test_idx = train_test_split(df.index,test_size=0.2, random_state=0)
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        return load_eval_dataset(train_df, test_df, tokenizer, args)   
    else:
        raise NotImplemented
