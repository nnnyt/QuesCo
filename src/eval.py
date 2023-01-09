from transformers import (
    AdamW,
    PreTrainedTokenizer,
    get_scheduler
)
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
import torch
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, classification_report, mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
import torch.nn as nn
import os


def eval_similarity(model, tokenizer: PreTrainedTokenizer, test_dataset, data_path, id_dict, args):

    def collate(examples: List[torch.Tensor]):
        ques = []
        for i in examples:
            ques.append(i["input_ids"])
        if tokenizer._pad_token is None:
            return pad_sequence(ques, batch_first=True)
        return pad_sequence(ques, batch_first=True, padding_value=tokenizer.pad_token_id)

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=8,
        pin_memory=False,
        worker_init_fn=lambda id: np.random.seed(torch.initial_seed() // 2 ** 32 + id),
        collate_fn=collate,
    )

    logging.info("[test-sim] inference questions")
    # inference
    model.eval()
    ques_scoring = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"[test-sim] inference questions"):
            batch = batch.to(args.device)
            ques_vec = model(batch).detach().cpu().numpy()
            ques_scoring.extend(ques_vec)
    ques_scoring = np.array(ques_scoring)

    logging.info(f'[test-sim] question scoring num: {ques_scoring.shape[0]}')

    # calculate similarity
    sim_path = data_path / 'similarity.csv'
    sim_df = pd.read_csv(sim_path)
    sim_df = sim_df[sim_df['id1'].isin(id_dict.keys())]
    sim_df = sim_df[sim_df['id2'].isin(id_dict.keys())]
    sim_df['id1'] = sim_df['id1'].apply(lambda x: ques_scoring[id_dict[x]])
    sim_df['id2'] = sim_df['id2'].apply(lambda x: ques_scoring[id_dict[x]])
    ques_emb1 = np.array(sim_df['id1'].tolist())
    ques_emb2 = np.array(sim_df['id2'].tolist())
    labels = np.array(sim_df['similarity'].tolist())

    cosine_scores = 1 - (paired_cosine_distances(ques_emb1, ques_emb2))
    manhattan_distances = -paired_manhattan_distances(ques_emb1, ques_emb2)
    euclidean_distances = -paired_euclidean_distances(ques_emb1, ques_emb2)
    dot_products = [np.dot(emb1, emb2) for emb1, emb2 in zip(ques_emb1, ques_emb2)]

    eval_pearson_cosine, _ = pearsonr(labels, cosine_scores)
    eval_spearman_cosine, _ = spearmanr(labels, cosine_scores)

    eval_pearson_manhattan, _ = pearsonr(labels, manhattan_distances)
    eval_spearman_manhattan, _ = spearmanr(labels, manhattan_distances)

    eval_pearson_euclidean, _ = pearsonr(labels, euclidean_distances)
    eval_spearman_euclidean, _ = spearmanr(labels, euclidean_distances)

    eval_pearson_dot, _ = pearsonr(labels, dot_products)
    eval_spearman_dot, _ = spearmanr(labels, dot_products)

    logging.info("[test-sim] Cosine-Similarity :\tPearson: {:.4f}\tSpearman: {:.4f}".format(
        eval_pearson_cosine, eval_spearman_cosine))
    logging.info("[test-sim] Manhattan-Distance:\tPearson: {:.4f}\tSpearman: {:.4f}".format(
        eval_pearson_manhattan, eval_spearman_manhattan))
    logging.info("[test-sim] Euclidean-Distance:\tPearson: {:.4f}\tSpearman: {:.4f}".format(
        eval_pearson_euclidean, eval_spearman_euclidean))
    logging.info("[test-sim] Dot-Product-Similarity:\tPearson: {:.4f}\tSpearman: {:.4f}".format(
        eval_pearson_dot, eval_spearman_dot))


def eval_classification(model, tokenizer: PreTrainedTokenizer, train_dataset, test_dataset, out_model_path, args):
    train_loader, test_loader = _set_loader(tokenizer, train_dataset, test_dataset, args)

    logging.info(f"[test-con] train concept predictor")
    if args.max_train_steps > 0:
        t_total = args.max_train_steps
        args.epochs = args.max_train_steps // len(train_loader) + 1
    else:
        t_total = len(train_loader) * args.epochs
    logging.info(f"[test-con] total steps: {t_total}")

    # Prepare optimizer and schedule
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = get_scheduler(
        "linear", optimizer=optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Train!
    logging.info("***** Running training *****")
    logging.info("  Num examples = %d", len(train_dataset))
    logging.info("  Num Epochs = %d", args.epochs)
    logging.info("  Instantaneous batch size = %d", args.batch_size)
    logging.info("  Total optimization steps = %d", t_total)

    model.train()
    global_step = 0
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(args.epochs):
        tr_loss = 0.0
        epoch_iterator = tqdm(train_loader, desc=f"[test-con] Epoch {epoch}")
        for step, batch in enumerate(epoch_iterator):
            optimizer.zero_grad()
            ques, label = batch
            ques = ques.to(args.device)
            label = label.to(args.device)
            output = model(ques)
            loss = loss_fn(output, label)
            loss.backward()
            tr_loss += loss.item()
            optimizer.step()
            scheduler.step()
            global_step += 1

            # validation
            if global_step % args.validation_steps == 0:
                logging.info('{0} epoch:{1}  iter:{2}/{3}  loss:{4}'.format("[test-con]", epoch, step, len(train_loader), tr_loss / step))
                accuracy, precision, recall, f1, auc = eval(model, test_loader, args)
                logging.info(f'[test-con] Accuracy: {accuracy:.4f}\tPrecision: {precision:.4f}\tRecall: {recall:.4f}\tF1: {f1:.4f}\tAUC: {auc:.4f}')

            # save
            if global_step % args.save_step == 0 and args.save_strategy == 'step':
                ckpt_path = os.path.join(out_model_path, "{}-{}".format('checkpoint', global_step))
                save_model(ckpt_path, model, optimizer, scheduler, args)

            if global_step > t_total:
                epoch_iterator.close()
                break

        logging.info('{0} epoch:{1}  loss:{2}'.format("[test-con]", epoch, tr_loss / len(train_loader)))
        accuracy, precision, recall, f1, auc = eval(model, test_loader, args, True)
        logging.info(f'[test-con] Accuracy: {accuracy:.4f}\tPrecision: {precision:.4f}\tRecall: {recall:.4f}\tF1: {f1:.4f}\tAUC: {auc:.4f}')

        if global_step > t_total:
            break
        if args.save_strategy == 'epoch':
            ckpt_path = os.path.join(out_model_path, "{}-{}".format('checkpoint', epoch))
            save_model(ckpt_path, model, optimizer, scheduler, args)

    accuracy, precision, recall, f1, auc = eval(model, test_loader, args)
    logging.info(f'[test-con] Accuracy: {accuracy:.4f}\tPrecision: {precision:.4f}\tRecall: {recall:.4f}\tF1: {f1:.4f}\tAUC: {auc:.4f}')
    ckpt_path = os.path.join(out_model_path, "{}".format('checkpoint'))
    save_model(ckpt_path, model, optimizer, scheduler, args)


def eval(model, test_loader, args, report=False, type='classification'):
    model.eval()
    pred = []
    label = []
    with torch.no_grad():
        for _, batch in enumerate(tqdm(test_loader, desc=f"[test] inference")):
            ques, target = batch
            ques = ques.to(args.device)
            output = model(ques)
            pred.extend(output.cpu().detach().numpy())
            label.extend(target.cpu().detach().numpy())
    model.train()
    if type == 'classification':
        return calculate_class_metrics(pred, label, report)
    elif type == 'regression':
        return calculate_regression_metrics(pred, label) 


def softmax(x):
    max = np.max(x,axis=-1,keepdims=True) #returns max of each row and keeps same dims
    e_x = np.exp(x - max) #subtracts each row with its max value
    sum = np.sum(e_x,axis=-1,keepdims=True) #returns sum of each row and keeps same dims
    f_x = e_x / sum 
    return f_x


def calculate_class_metrics(pred, truth, report=False):
    pred = np.array(pred)
    truth = np.array(truth)
    y_pred = np.argmax(pred, axis=-1)
    pred_softmax = softmax(pred)
    accuracy = np.sum(y_pred == truth) / len(y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(truth, y_pred, average='macro')
    try:
        auc = roc_auc_score(truth, pred_softmax, multi_class='ovo')
    except ValueError:
        auc = 0
    if report:
        logging.info(classification_report(truth, y_pred))
    return accuracy, precision, recall, f1, auc


def calculate_regression_metrics(pred, truth):
    # (#samples)
    pred = np.array(pred).squeeze()
    truth = np.array(truth).squeeze()
    mae = mean_absolute_error(truth, pred)
    rmse = np.sqrt(mean_squared_error(truth, pred))
    p_corr = pearsonr(truth, pred)[0]
    sp_corr = spearmanr(truth, pred)[0]
    doa = calculate_doa(truth, pred)
    return mae, rmse, p_corr, sp_corr, doa


def calculate_doa(truth, pred):
    doa_score = 0.0
    all = 0.0
    for i in range(truth.shape[0]):
        for j in range(truth.shape[0]):
            if truth[i] > truth[j]:
                all += 1
                if pred[i] > pred[j]:
                    doa_score += 1
    return doa_score / all


def save_model(ckpt_path, model, optimizer, scheduler, args):
    os.makedirs(ckpt_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(ckpt_path, "model.pt"))
    torch.save(args, os.path.join(ckpt_path, "training_args.bin"))
    logging.info(f"[test-con] Saving model checkpoint to %s", ckpt_path)

    torch.save(optimizer.state_dict(), os.path.join(ckpt_path, "optimizer.pt"))
    torch.save(scheduler.state_dict(), os.path.join(ckpt_path, "scheduler.pt"))
    logging.info(f"[test-con] Saving optimizer and scheduler states to %s", ckpt_path)


def eval_regression(model, tokenizer: PreTrainedTokenizer, train_dataset, test_dataset, out_model_path, args):
    train_loader, test_loader = _set_loader(tokenizer, train_dataset, test_dataset, args)

    logging.info(f"[test-diff] train difficulty predictor")
    if args.max_train_steps > 0:
        t_total = args.max_train_steps
        args.epochs = args.max_train_steps // len(train_loader) + 1
    else:
        t_total = len(train_loader) * args.epochs
    logging.info(f"[test-diff] total steps: {t_total}")

    # Prepare optimizer and schedule
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = get_scheduler(
        "linear", optimizer=optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Train!
    logging.info("***** Running training *****")
    logging.info("  Num examples = %d", len(train_dataset))
    logging.info("  Num Epochs = %d", args.epochs)
    logging.info("  Instantaneous batch size = %d", args.batch_size)
    logging.info("  Total optimization steps = %d", t_total)

    model.train()
    global_step = 0
    loss_fn = nn.MSELoss()
    for epoch in range(args.epochs):
        tr_loss = 0.0
        epoch_iterator = tqdm(train_loader, desc=f"[test-diff] Epoch {epoch}")
        for step, batch in enumerate(epoch_iterator):
            optimizer.zero_grad()
            ques, label = batch
            ques = ques.to(args.device)
            label = label.to(args.device)
            output = model(ques)
            loss = loss_fn(output.squeeze(), label)
            loss.backward()
            tr_loss += loss.item()
            optimizer.step()
            scheduler.step()
            global_step += 1

            # validation
            if global_step % args.validation_steps == 0:
                logging.info('{0} epoch:{1}  iter:{2}/{3}  loss:{4}'.format("[test-diff]", epoch, step, len(train_loader), tr_loss / step))
                mae, rmse, p_corr, sp_corr, doa = eval(model, test_loader, args, type='regression')
                logging.info(f'[test-diff] MAE: {mae:.4f}\tRMSE: {rmse:.4f}\tPearson: {p_corr:.4f}\tSpearman: {sp_corr:.4f}\tDOA: {doa:.4f}')

            # save
            if global_step % args.save_step == 0 and args.save_strategy == 'step':
                ckpt_path = os.path.join(out_model_path, "{}-{}".format('checkpoint', global_step))
                save_model(ckpt_path, model, optimizer, scheduler, args)

            if global_step > t_total:
                epoch_iterator.close()
                break

        logging.info('{0} epoch:{1}  loss:{2}'.format("[test-diff]", epoch, tr_loss / len(train_loader)))
        mae, rmse, p_corr, sp_corr, doa = eval(model, test_loader, args, type='regression')
        logging.info(f'[test-diff] MAE: {mae:.4f}\tRMSE: {rmse:.4f}\tPearson: {p_corr:.4f}\tSpearman: {sp_corr:.4f}\tDOA: {doa:.4f}')

        if global_step > t_total:
            break
        if args.save_strategy == 'epoch':
            ckpt_path = os.path.join(out_model_path, "{}-{}".format('checkpoint', epoch))
            save_model(ckpt_path, model, optimizer, scheduler, args)

    mae, rmse, p_corr, sp_corr, doa = eval(model, test_loader, args, type='regression')
    logging.info(f'[test-diff] MAE: {mae:.4f}\tRMSE: {rmse:.4f}\tPearson: {p_corr:.4f}\tSpearman: {sp_corr:.4f}\tDOA: {doa:.4f}')
    ckpt_path = os.path.join(out_model_path, "{}".format('checkpoint'))
    save_model(ckpt_path, model, optimizer, scheduler, args)


def _set_loader(tokenizer, train_dataset, test_dataset, args):
    def collate(examples):
        ques, label = [], []
        ques = []
        for i in examples:
            ques.append(i["input_ids"])
            label.append(i["label"])
        if tokenizer._pad_token is None:
            return pad_sequence(ques, batch_first=True), torch.tensor(label)
        return pad_sequence(ques, batch_first=True, padding_value=tokenizer.pad_token_id), \
            torch.tensor(label)

    train_loader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=8,
        pin_memory=False,
        worker_init_fn=lambda id: np.random.seed(torch.initial_seed() // 2 ** 32 + id),
        collate_fn=collate,
    )

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.valid_batch_size if args.valid_batch_size > 0 else args.batch_size,
        drop_last=False,
        num_workers=8,
        pin_memory=False,
        worker_init_fn=lambda id: np.random.seed(torch.initial_seed() // 2 ** 32 + id),
        collate_fn=collate,
    )
    return train_loader, test_loader
