from re import I
import torch
from torch import nn
from transformers import AutoModel


class MoCoTemplate(nn.Module):
    """From https://github.com/facebookresearch/moco/blob/master/moco/builder.py"""

    def __init__(self, d_rep=128, K=61440, m=0.999, T=0.07, encoder_params={}):  # 61440 = 2^12 * 3 * 5
        """
        d_rep: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super().__init__()
        self.config = dict(**{"moco_num_keys": K, "moco_momentum": m, "moco_temperature": T}, **encoder_params)
        self.K = K
        self.m = m
        self.T = T

        self.encoder_q = self.make_encoder(**encoder_params)
        self.encoder_k = self.make_encoder(**encoder_params)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        # for param_k in self.encoder_k.parameters():
        #     param_k.requires_grad = False

        self.register_buffer("queue", torch.randn(d_rep, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("label_queue", torch.ones(K, dtype=torch.long) * -1)


    def make_encoder(self, **kwargs):
        raise NotImplementedError()

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        # ptr = int(self.queue_ptr)
        ptr = int(self.queue_ptr.item())
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue = torch.cat([self.queue[:, :ptr], keys.T, self.queue[:, ptr + batch_size :]], dim=1).detach()
        self.label_queue = torch.cat([self.label_queue[:ptr], labels, self.label_queue[ptr + batch_size : ]]).detach()
        # self.queue = self.queue.clone()
        # self.queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def embed_x(self, img, lens):
        return self.encoder_q(img, lens)

    def forward(self, im_q, im_k, labels=None):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
            lengths_k: sequence length, [B,]
            lengths_q: sequence length, [B,]
            q: queries, pre-computed embedding of im_q, [N, C]
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", *[q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", *[q, self.queue.detach()])


        # dequeue and enqueue
        # print("world size", torch.distributed.get_world_size())
        self._dequeue_and_enqueue(k, labels)

        # return logits, targets
        return l_pos, l_neg, labels, self.label_queue


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


class QuesEncoder(nn.Module):
    def __init__(
        self,
        bert_type,
        d_rep=256,
        project=False,
    ):
        super().__init__()
        self.config = {k: v for k, v in locals().items() if k != "self"}
        self.encoder = AutoModel.from_pretrained(bert_type)
        if project:
            self.project_layer = nn.Sequential(nn.Linear(self.encoder.config.hidden_size,
                                                    self.encoder.config.hidden_size),
                                          nn.ReLU(),
                                          nn.Linear(self.encoder.config.hidden_size, d_rep))
        # NOTE: We use the default PyTorch intialization, so no need to reset parameters.

    def forward(self, x, lengths=None, no_project_override=False):
        # x: (batch_size, max_len)
        out = self.encoder(x).last_hidden_state
        if not no_project_override and self.config["project"]:
            out = self.project(out)
            # out: (batch_size, question_dim)
            return out

        # (batch_size, max_len, question_dim)
        return out, None

    def project(self, out):
        # out: (batch_size, max_len, hidden_dim)
        assert self.config["project"]
        # NOTE: This computes a mean pool of the token representations across ALL tokens,
        # including padding from uneven lengths in the batch.
        # (batch_size, question_dim)
        return self.project_layer(out.mean(dim=1))


class QuesMoCo(MoCoTemplate):
    def __init__(self, bert_type, d_rep=128, project=True, K=107520, m=0.999, T=0.07, encoder_config={}):
        super().__init__(
            d_rep,
            K,
            m,
            T,
            encoder_params=dict(bert_type=bert_type, d_rep=d_rep, project=project, **encoder_config),
        )

    def make_encoder(
        self,
        bert_type,
        d_rep,
        project=True,
        **kwargs
    ):
        return QuesEncoder(
            bert_type, project=project, d_rep=d_rep, **kwargs
        )

    def forward(self, im_q, im_k, labels=None):
        """
        Input:
            im_q: a batch of query images, [batch_size, max_len]
            im_k: a batch of key images, [batch_size, max_len]
        Output:
            logits, targets
        """
        return super().forward(im_q, im_k, labels=labels)


class Predictor(nn.Module):
    def __init__(self, bert_type, d_rep, pad_id, n_classes=0, classifier=False, finetune_all=False, encoder_config={}):
        super().__init__()
        self.config = {k: v for k, v in locals().items() if k != "self"}
        self.encoder = QuesEncoder(
            bert_type, project=False, d_rep=d_rep, **encoder_config
        )
        if classifier:
            hidden_size = self.encoder.encoder.config.hidden_size
            if n_classes == 1:
                # regression
                self.classifier = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(),
                    nn.Linear(hidden_size // 2, n_classes), nn.Sigmoid()
                )
            else:
                self.classifier = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(),
                    nn.Linear(hidden_size // 2, n_classes))

    def forward(self, ids, lengths=None):
        # ids: (batch_size, max_len)

        # NOTE: zero-shot similarity prediction, only get transformers output here
        # (batch_size, max_len, hidden_size)
        if not self.config['finetune_all']:
            with torch.no_grad():
                emb, _ = self.encoder(ids, lengths)
        else:
            emb, _ = self.encoder(ids, lengths)

        # Pooling of non-padding words
        # (batch_size, max_len)
        non_padding_mask = ids != self.config["pad_id"]
        # (batdh_size, 1)
        num_non_padding = non_padding_mask.sum(dim=1).unsqueeze(-1)
        # (batch_size, max_len, 1)
        non_padding_mask = non_padding_mask.unsqueeze(-1)  # [L, 2B, 1]

        # Mean pooling
        emb = emb * non_padding_mask
        # (batch_size, hidden_size)
        emb = emb.sum(dim=1) / num_non_padding.float()
        # Max pool
        # emb = emb * non_padding_mask + emb.min() * ~non_padding_mask
        # emb, _ = emb.max(dim=1)

        # for concept/difficulty prediction
        if self.config['classifier']:
            if not self.config['finetune_all']:
                # stop gradient propagation to encoder
                emb = emb.detach()
            # (batch_size, n_classes)
            return self.classifier(emb)

        # (batch_size, hidden_size)
        return emb
