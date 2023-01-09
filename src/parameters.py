import os
import sys
import logging
import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = argparse.ArgumentParser()
    # data configuration
    parser.add_argument(
        "--data_path",
        type=str,
        default=os.getenv("AMLT_DATA_DIR", "../data"),
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default=os.getenv("AMLT_OUTPUT_DIR", "../output"),
    )
    parser.add_argument(
        "--data",
        type=str,
        default="all",
        choices=["SYSTEM1", "SYSTEM2", "all", "examples"],
    )
    parser.add_argument("--data_filename", type=str, default="pretrain_question.csv", help="the filename of training data")

    # job configuration
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--name", type=str, default="QuesCo", help="job name.")

    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--valid_batch_size", type=int, default=0)
    parser.add_argument("--max_train_steps", type=int, default=0)
    parser.add_argument("--save_step", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--validation_steps", type=int, default=500)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--n_gpu", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_strategy", type=str, default="step", choices=['epoch', 'step'])
    parser.add_argument("--gradient_accumulate_step", type=int, default=1)

    # model configuration
    parser.add_argument('--bert_type', type=str, default="bert-base-chinese")
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--add_latex", type=str2bool, default=False, help="whether to add latex symbols to tokenizer")
    parser.add_argument("--lower", type=str2bool, default=False)

    # train
    parser.add_argument("--augmentation_strategy", type=str, default="all",
                        choices=["all", "random"],
                        help="The data augmentation strategy in contrastive learning")
    parser.add_argument("--text_augment", type=str2bool, default=True, help="whether to do augmentation for text")
    parser.add_argument("--formula_augment", type=str2bool, default=True, help="whether to do augmentation for formulas")
    parser.add_argument("--ques_augment", type=str2bool, default=True, help='whether to do augmentation for questions')
    parser.add_argument("--project", type=str2bool, default=True, help="whether to add a projector after transformer")
    parser.add_argument("--ques_dim", type=int, default=768, help="the dimension of the question representation")
    parser.add_argument("--queue_size", type=int, default=1000, help="the size of the memory queue")
    parser.add_argument("--momentum", type=float, default=0.999, help="moco momentum of updating key encoder")
    parser.add_argument("--temprature", type=float, default=0.07, help="softmax temperature of Moco")
    parser.add_argument("--augment_prob", type=float, default=0.7, help="the propability of applying one augmentation")
    # ranking loss
    parser.add_argument("--rank", type=str2bool, default=True, help="")
    parser.add_argument("--similarity_threshold", type=float, default=0.01, help="")
    parser.add_argument("--n_sim_classes", type=int, default=10, help="only use top n similar classes")
    parser.add_argument('--min_tau', default=0.1, type=float, help='min temperature parameter in SimCLR')
    parser.add_argument('--max_tau', default=0.2, type=float, help='max temperature parameter in SimCLR')
    parser.add_argument('--one_loss_per_rank', type=str2bool, default=True, help="")
    parser.add_argument('--mixed_out_in', type=str2bool, default=False)
    parser.add_argument('--do_sum_in_log', type=str2bool, default=True)
    parser.add_argument('--use_same_and_similar_class', type=str2bool, default=False, help='')
    parser.add_argument("--use_data_augmentation", type=str2bool, default=True)
    parser.add_argument("--unsup_tau", default=0.07, type=float)

    # test tasks
    parser.add_argument("--test_task", type=str, default=None, choices=["similarity", "concepts", "difficulty"], help="test task")
    parser.add_argument("--test_know_level", type=int, default=1, choices=[1, 2, 3, 4], help="choose which knowledge level to evaluate")
    parser.add_argument("--finetune_all_model", type=str2bool, default=False, help="Whether to finetune the whole model(including encoder)")

    args = parser.parse_args()
    return args


def setuplogger(args, out_path):
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(levelname)s %(asctime)s] %(message)s")
    handler.setFormatter(formatter)
    root.addHandler(handler)

    fh = logging.FileHandler(out_path / f"log.{args.mode}.txt")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    root.addHandler(fh)
