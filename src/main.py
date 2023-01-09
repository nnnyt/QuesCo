import torch
import os
import logging
from pathlib import Path
from parameters import parse_args, setuplogger
from utils import set_seed
from transformers import (
    # AutoConfig,
    AutoTokenizer,
)
import json
from model import QuesMoCo, Predictor
from dataset import load_dataset, load_test_dataset
from train import train
from eval import eval_classification, eval_similarity, eval_regression


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parse_args()
    out_path = Path(args.out_path) / f"{args.name}-{args.data}"
    out_path.mkdir(exist_ok=True, parents=True)

    setuplogger(args, out_path)
    logging.info(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    # device = torch.device("cpu")
    args.device = device

    set_seed(args)

    if args.mode == 'train':
        data_path = Path(args.data_path) / args.data / args.data_filename
        cache_path = Path(args.data_path) / args.data / 'cache'
        out_path = Path(args.out_path) / f"{args.name}-{args.data}"
        out_model_path = out_path / "model"

        logging.info("[train] begin training")
        logging.info("[train] loading model")
        model = QuesMoCo(args.bert_type, args.ques_dim, args.project, args.queue_size, args.momentum)
        model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(args.bert_type, model_max_length=args.max_len)
        if args.add_latex:
            logging.info("add latex symbols into tokenizer and model")
            with open('./latex_thresh.json', 'r') as f:
                latex_symbols = json.load(f)
            new_tokens = [w for w in latex_symbols if w not in tokenizer.vocab]
            if new_tokens is not None:
                logging.info(f"found {len(new_tokens)} new tokens")
                tokenizer.add_tokens(new_tokens)
                model.encoder_q.encoder.resize_token_embeddings(len(tokenizer))
                model.encoder_k.encoder.resize_token_embeddings(len(tokenizer))

        logging.info("[train] loading data")
        train_dataset = load_dataset(data_path, tokenizer, cache_path, args)["train"]
        train(model, tokenizer, train_dataset, out_model_path, args)

    elif args.mode == 'test':
        assert args.test_task, "test_task must be specified in test mode"
        data_path = Path(args.data_path) / args.data / 'processed'
        cache_path = Path(args.data_path) / args.data / 'cache'
        out_path = Path(args.out_path) / f"{args.name}-{args.data}"
        logging.info("[test] begin test")
        if args.test_task == "similarity":
            logging.info("[test-sim] task: similarity prediction")
            tokenizer = AutoTokenizer.from_pretrained(args.bert_type, model_max_length=args.max_len)
            test_dataset, id_dict = load_test_dataset(data_path, tokenizer, args)
            model = Predictor(args.bert_type, args.ques_dim, tokenizer.pad_token_id)
            model.to(device)
            eval_similarity(model, tokenizer, test_dataset, data_path, id_dict, args)
        elif args.test_task == "concepts":
            logging.info("[test-con] task: concepts prediction")
            out_model_path = out_path / "concepts_model"
            tokenizer = AutoTokenizer.from_pretrained(args.bert_type, model_max_length=args.max_len)
            dataset, n_classes = load_test_dataset(data_path, tokenizer, args)
            model = Predictor(args.bert_type, args.ques_dim, tokenizer.pad_token_id, n_classes, True, args.finetune_all_model)
            model.to(device)
            eval_classification(model, tokenizer, dataset["train"], dataset["test"], out_model_path, args)
        elif args.test_task == "difficulty":
            logging.info("[test-diff] task: difficulty prediction")
            out_model_path = out_path / "difficulty_model"
            tokenizer = AutoTokenizer.from_pretrained(args.bert_type, model_max_length=args.max_len)
            dataset = load_test_dataset(data_path, tokenizer, args)
            model = Predictor(args.bert_type, args.ques_dim, tokenizer.pad_token_id, 1, True, args.finetune_all_model)
            model.to(device)
            eval_regression(model, tokenizer, dataset["train"], dataset["test"], out_model_path, args)
        else:
            raise NotImplementedError
