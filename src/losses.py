import torch.nn as nn
import torch
import pandas as pd
import torch.nn.functional as F
import logging
from tqdm import tqdm


class ContrastiveRankingLoss(object):
    def __init__(self, data_path, args):
        self.similar_knowledge, self.class_num = self.process_knowledge(data_path)

        self.similarity_threshold = args.similarity_threshold
        self.n_sim_classes = args.n_sim_classes
        self.use_all_ranked_classes_above_threshold = self.similarity_threshold > 0
        self.min_tau = args.min_tau
        self.max_tau = args.max_tau
        self.one_loss_per_rank = args.one_loss_per_rank
        self.mixed_out_in = args.mixed_out_in
        self.do_sum_in_log = args.do_sum_in_log
        self.use_same_and_similar_class = args.use_same_and_similar_class
        self.args = args

        self.cross_entropy = nn.CrossEntropyLoss()

    def process_knowledge(self, data_path):
        know_df = pd.read_csv(data_path)
        similar_knowledge = {}
        for _, r in tqdm(know_df.iterrows(), "[Loss] processing knowledge hierarchy"):
            level3_sim_classes = [r.level_3_knowledge]
            level2_sim_classes = know_df[know_df['level_2_knowledge'] == r.level_2_knowledge]['level_3_knowledge'].unique().tolist()
            level1_sim_classes = know_df[know_df['level_1_knowledge'] == r.level_1_knowledge]['level_3_knowledge'].unique().tolist()
            for c in level2_sim_classes:
                level1_sim_classes.remove(c)
            for c in level3_sim_classes:
                level2_sim_classes.remove(c)
            sim_classes = level3_sim_classes + level2_sim_classes + level1_sim_classes
            other_classes = know_df[~know_df['level_3_knowledge'].isin(sim_classes)]['level_3_knowledge'].unique().tolist()
            # 所有知识点，当前知识点+二级相同的知识点+一级相同的知识点+不同的知识点
            sim_classes = sim_classes + other_classes
            similar_knowledge[r.level_3_knowledge] = {
                'sim_class': torch.tensor(sim_classes).long(),
                'sim_class_weight': torch.cat(
                [torch.ones((len(level3_sim_classes), 1), dtype=torch.float32),
                 torch.ones((len(level2_sim_classes), 1), dtype=torch.float32) * 0.75,
                 torch.ones((len(level1_sim_classes), 1), dtype=torch.float32) * 0.5,
                 torch.zeros((len(other_classes), 1), dtype=torch.float32)], dim=0).squeeze()
            }
        return similar_knowledge, len(sim_classes)

    def sum_in_log(self, l_pos, l_neg, tau):
        logits = torch.cat([l_pos, l_neg], dim=1) / tau
        logits = F.softmax(logits, dim=1)
        sum_pos = logits[:, 0:l_pos.shape[1]].sum(1)
        sum_pos = sum_pos[sum_pos > 1e-7]
        if len(sum_pos) > 0:
            loss = - torch.log(sum_pos).mean()
        else:
            loss = torch.tensor([0.0]).to(self.args.device)
        return loss

    def sum_out_log(self, l_pos, l_neg, tau):
        l_pos = l_pos / tau
        l_neg = l_neg / tau
        l_pos_exp = torch.exp(l_pos)
        l_neg_exp_sum = torch.exp(l_neg).sum(dim=1).unsqueeze(1)
        all_scores = (l_pos_exp / (l_pos_exp + l_neg_exp_sum))
        all_scores = all_scores[all_scores > 1e-7]
        if len(all_scores) > 0:
            loss = - torch.log(all_scores).mean()
        else:
            loss = torch.tensor([0.0]).to(self.args.device)
        return loss

    def get_similar_labels(self, labels):
        # in this case use top n classes
        labels = labels.cpu().numpy()

        sim_class_labels = torch.zeros(
            (labels.shape[0], self.class_num)).type(torch.long).to(self.args.device)
        sim_class_sims = torch.zeros(
            (labels.shape[0], self.class_num)).type(torch.float).to(self.args.device)
        sim_leq_thresh = torch.zeros(
            (labels.shape[0], self.class_num)).type(torch.bool).to(self.args.device)
        for i, label in enumerate(labels):
            sim_class_labels[i, :] = self.similar_knowledge[label]['sim_class']
            sim_class_sims[i, :] = self.similar_knowledge[label]['sim_class_weight']
            sim_leq_thresh[i, :] = self.similar_knowledge[label]['sim_class_weight'] >= self.similarity_threshold
        # remove columns in which no sample has a similarity  qual to or larger than the selected threshold
        at_least_one_leq_thrsh = torch.sum(sim_leq_thresh, dim=0) > 0
        sim_class_labels = sim_class_labels[:, at_least_one_leq_thrsh]
        sim_leq_thresh = sim_leq_thresh[:, at_least_one_leq_thrsh]

        sim_class_labels = sim_class_labels[:, :self.n_sim_classes]
        sim_class_sims = sim_class_sims[:, :self.n_sim_classes]

        # negate sim_leq_thresh to get a mask that can be applied to set all values below thresh to -inf
        sim_leq_thresh = ~sim_leq_thresh[:, :self.n_sim_classes]
        # sim_leq_thresh, bool, 如果是正样本就是False，否则是True
        # (bz, n_sim_classes), (bz, n_sim_classes), (bz, n_sim_classes)
        return sim_class_labels, sim_leq_thresh, sim_class_sims

    def get_dynamic_tau(self, similarities):
        dissimilarities = 1 - similarities
        d_taus = self.min_tau + (dissimilarities - 0) / (1 - 0) * (self.max_tau - self.min_tau)
        return d_taus

    def compute_InfoNCE_classSimilarity(self, l_pos, l_neg, labels, label_queue):
        """
        l_pos: (bz, 1)
        l_neg: (bz, K)
        labels: (bz)
        label_queue(K)
        """
        similar_labels, below_threshold, class_sims = self.get_similar_labels(labels)
        masks = []
        threshold_masks = []
        dynamic_taus = []
        for i in range(similar_labels.shape[1]):
            # (bz, K) bool 对每个sample的top i similar label，queue中每个样本是否相等
            mask = (label_queue[:, None] == similar_labels[None, :, i]).transpose(0, 1)
            masks.append(mask)
            if self.use_all_ranked_classes_above_threshold:
                # (bz, K) 如果是负样本则一行都是False 否则为True
                threshold_masks.append(below_threshold[None, :, i].transpose(0, 1).repeat(1, mask.shape[1]))
            dynamic_taus.append(self.get_dynamic_tau(class_sims[:, i]))

        if self.one_loss_per_rank:
            # 一共几种score
            similarity_scores = reversed(class_sims.unique(sorted=True))
            similarity_scores = similarity_scores[similarity_scores > -1]
            new_masks = []
            new_taus = []
            for s in similarity_scores:
                new_taus.append(self.get_dynamic_tau(torch.ones_like(dynamic_taus[0]) * s))
                mask_all_siblings = torch.zeros_like(masks[0], dtype=torch.bool)
                for i in range(similar_labels.shape[1]):
                    same_score = class_sims[:, i] == s
                    if any(same_score):
                        mask_all_siblings[same_score] = mask_all_siblings[same_score] | masks[i][same_score]
                new_masks.append(mask_all_siblings)
            masks = new_masks
            dynamic_taus = new_taus

        l_class_pos = l_neg.clone()

        return l_pos, l_class_pos, l_neg, masks, threshold_masks, dynamic_taus

    def __call__(self, l_pos, l_neg, labels, label_queue):
        """
        l_pos: (bz, 1)
        l_neg: (bz, K)
        labels: (bz)
        label_queue(K)
        """
        l_pos, l_class_pos, l_neg, masks, below_threshold, dynamic_taus = self.compute_InfoNCE_classSimilarity(l_pos, l_neg, labels, label_queue)

        #initially l_neg and l_class pos are identical
        res = {}

        if self.args.use_data_augmentation:
            # augmentation sample > all other samples
            res['class_similarity_ranking_class'] = {
                'score': None,
                'target': None,
                'loss': self.sum_out_log(l_pos, l_neg, dynamic_taus[0].view(-1, 1))
            }

        for i, mask in enumerate(masks):
            if (self.use_same_and_similar_class and not i == 0):
                mask = masks[-1]
                for j in range(len(masks)-1):
                    mask = mask | masks[j]
                l_neg[mask & ~below_threshold[i]] = -float("inf")
                l_class_pos_cur = l_class_pos.clone()
                #keep only members of current class
                l_class_pos_cur[~mask] = -float("inf")
                # throw out those batches for which the similarity between ranking class and label class is below threshold
                l_class_pos_cur[below_threshold[i]] = -float("inf")

            elif self.use_all_ranked_classes_above_threshold or (self.use_same_and_similar_class and i == 0):
                # mask out from negatives only if they are part of the class and this class has a similarity to
                # label class above the similarity threshold
                l_neg[mask & ~below_threshold[i]] = -float("inf")
                l_class_pos_cur = l_class_pos.clone()
                l_class_pos_cur[~mask] = -float("inf")
                l_class_pos_cur[below_threshold[i]] = -float("inf")

            else:
                l_neg[mask] = -float("inf")
                l_class_pos_cur = l_class_pos.clone()
                l_class_pos_cur[~mask] = -float("inf")
            taus = dynamic_taus[i].view(-1, 1)

            # if i == 0:
            #     #l_class_por_cur: (bz, K) -> (bz, 1+K)
            #     l_class_pos_cur = torch.cat([l_pos, l_class_pos_cur], dim=1)

            if self.mixed_out_in and i == 0:
                loss = self.sum_out_log(l_class_pos_cur, l_neg, taus)
            elif self.do_sum_in_log and not(self.mixed_out_in and i ==0):
                loss = self.sum_in_log(l_class_pos_cur, l_neg, taus)
            else:
                loss = self.sum_out_log(l_class_pos_cur, l_neg, taus)
            result = {'score': None,
                      'target': None,
                      'loss': loss}
            res['class_similarity_ranking_class' + str(i)] = result

            if (self.use_same_and_similar_class and not i == 0):
                break

        return self.criterion(res)

    def criterion(self, outputs):
        loss = 0.0
        for key, val in outputs.items():
            if 'loss' in val:
                loss = loss + val['loss']
            else:
                loss = loss + self.cross_entropy(val['score'], val['target'])
        loss = loss / float(len(outputs))
        return loss
