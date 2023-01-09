from transformers import (
    AdamW,
    PreTrainedTokenizer,
    get_scheduler
)
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, RandomSampler
import torch
import logging
from tqdm import tqdm
from torch.nn import functional as F
import os
from pathlib import Path
import numpy as np
from losses import ContrastiveRankingLoss


def train(model, tokenizer: PreTrainedTokenizer, train_dataset, out_model_path, args):

    def collate(examples):
        ques_q, ques_k, labels = [], [], []
        for i in examples:
            ques_q.append(i["input_ids"])
            ques_k.append(i["aug-input_ids"])
            labels.append(i["level_3_knowledge"])
        if tokenizer._pad_token is None:
            return pad_sequence(ques_q, batch_first=True), pad_sequence(ques_k, batch_first=True),torch.tensor(labels)
        return pad_sequence(ques_q, batch_first=True, padding_value=tokenizer.pad_token_id), \
            pad_sequence(ques_k, batch_first=True, padding_value=tokenizer.pad_token_id), \
                torch.tensor(labels)

    train_loader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=args.batch_size,
        drop_last=True, # drop_last=False 无法保证MoCo中 self.k % batch_size == 0
        num_workers=8,
        pin_memory=False,
        worker_init_fn=lambda id: np.random.seed(torch.initial_seed() // 2 ** 32 + id),
        collate_fn=collate,
    )

    if args.max_train_steps > 0:
        t_total = args.max_train_steps
        args.epochs = args.max_train_steps // len(train_loader) + 1
    else:
        t_total = len(train_loader) * args.epochs
    logging.info(f"[CL] total steps: {t_total}")
    # model.resize_token_embeddings(len(tokenizer))

    # Prepare optimizer and schedule
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = get_scheduler(
        "linear", optimizer=optimizer, num_warmup_steps=args.warmup_steps,
        num_training_steps=t_total // args.gradient_accumulate_step
    )

    # loss
    loss_data = Path(args.data_path) / args.data / 'processed' / 'knowledge.csv'
    if args.rank:
        criterion = ContrastiveRankingLoss(loss_data, args)
    else:
        criterion = F.cross_entropy

    # Train!
    logging.info("***** Running training *****")
    logging.info("  Num examples = %d", len(train_dataset))
    logging.info("  Num Epochs = %d", args.epochs)
    logging.info("  Instantaneous batch size = %d", args.batch_size)
    logging.info("  Total optimization steps = %d", t_total)
    logging.info("  Gradient accumulation step = %d", args.gradient_accumulate_step)

    model.train()
    global_step = 0
    for epoch in range(args.epochs):
        tr_loss = 0.0
        epoch_iterator = tqdm(train_loader, desc=f"[CL] Epoch {epoch}")
        for step, batch in enumerate(epoch_iterator):
            ques_q, ques_k, labels = batch
            ques_q = ques_q.to(args.device)
            ques_k = ques_k.to(args.device)
            labels = labels.to(args.device)
            l_pos, l_neg, labels, label_queue = model(ques_q, ques_k, labels)
            if args.rank:
                loss = criterion(l_pos, l_neg, labels, label_queue)
            else:
                logits = torch.cat([l_pos, l_neg], dim=1)
                logits /= args.min_tau
                labels = torch.zeros(logits.shape[0], dtype=torch.long).to(l_pos.device)
                loss = criterion(logits, labels)
            loss = loss / args.gradient_accumulate_step
            loss.backward()
            tr_loss += loss.item()
            if ((step + 1) % args.gradient_accumulate_step == 0) or (args.gradient_accumulate_step + 1 == len(epoch_iterator)):
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
            global_step += 1

            # validation
            if global_step % args.validation_steps == 0:
                logging.info('[CL] epoch:{0}  iter:{1}/{2}  loss:{3}'.format(epoch, step, len(train_loader), tr_loss / step))
                # TODO: validation

            # save
            if global_step % args.save_step == 0 and args.save_strategy == 'step':
                ckpt_path = os.path.join(out_model_path, "{}-{}-{}".format('cl', 'checkpoint', global_step))
                save_model(ckpt_path, model, tokenizer, optimizer, scheduler, args)

            if global_step > t_total:
                epoch_iterator.close()
                break
        if global_step > t_total:
            break
        if args.save_strategy == 'epoch':
            ckpt_path = os.path.join(out_model_path, "{}-{}-{}".format('cl', 'checkpoint', epoch))
            save_model(ckpt_path, model, tokenizer, optimizer, scheduler, args)
    ckpt_path = os.path.join(out_model_path, "{}-{}".format('cl', 'checkpoint'))
    save_model(ckpt_path, model, tokenizer, optimizer, scheduler, args)


def save_model(ckpt_path, model, tokenizer, optimizer, scheduler, args):
    os.makedirs(ckpt_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(ckpt_path, "model.pt"))
    tokenizer.save_pretrained(ckpt_path)
    torch.save(args, os.path.join(ckpt_path, "training_args.bin"))
    model.encoder_q.encoder.save_pretrained(ckpt_path)
    logging.info("[CL] Saving model checkpoint to %s", ckpt_path)

    torch.save(optimizer.state_dict(), os.path.join(ckpt_path, "optimizer.pt"))
    torch.save(scheduler.state_dict(), os.path.join(ckpt_path, "scheduler.pt"))
    logging.info("[CL] Saving optimizer and scheduler states to %s", ckpt_path)
