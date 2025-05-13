# import faiss
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import math
import copy
import numpy as np
from domainbed import networks
from domainbed.lib.misc import random_pairs_of_minibatches
from torch.cuda.amp import autocast

ALGORITHMS = [
    "ERM",
    "KNN",
    "IRM",
    "DRM",
    "GroupDRO",
    "Mixup",
    "MLDG",
    "CORAL",
    "MMD",
    "DANN",
    "CDANN",
    "MTL",
    "SagNet",
    "ARM",
    "VREx",
    "RSC",
    "SD",
    "IBIRM",
]


class MomentumQueueClass(nn.Module):
    def __init__(self, feature_dim, queue_size, temperature, k, classes):
        super(MomentumQueue, self).__init__()
        self.queue_size = queue_size
        self.index = 0
        self.temperature = temperature
        self.k = k
        self.classes = classes

        # noinspection PyCallingNonCallable
        self.register_buffer("params", torch.tensor([-1]))
        stdv = 1.0 / math.sqrt(feature_dim / 3)
        memory = (
            torch.rand(self.queue_size, feature_dim, requires_grad=False)
            .mul_(2 * stdv)
            .add_(-stdv)
        )
        self.register_buffer("memory", memory)
        memory_label = torch.zeros(self.queue_size).long()
        self.register_buffer("memory_label", memory_label)

    def update_queue(self, k_all, k_label_all):
        with torch.no_grad():
            k_all = F.normalize(k_all)
            all_size = k_all.shape[0]
            out_ids = torch.fmod(
                torch.arange(all_size, dtype=torch.long).cuda() + self.index,
                self.queue_size,
            )
            self.memory.index_copy_(0, out_ids, k_all)
            self.memory_label.index_copy_(0, out_ids, k_label_all)
            self.index = (self.index + all_size) % self.queue_size

    def forward(self, x, test=False):
        dist = torch.mm(
            F.normalize(x), self.memory.transpose(1, 0)
        )  # B * Q, memory already normalized
        # compute cos similarity between each feature vector and feature bank ---> [B, N]
        sim_weight, sim_indices = torch.topk(dist, k=self.k)
        sim_labels = torch.gather(
            self.memory_label.expand(x.size(0), -1), dim=-1, index=sim_indices
        )
        # sim_weight = (sim_weight / self.temperature).exp()
        if not test:
            sim_weight = F.softmax(sim_weight / self.temperature, dim=1)
        else:
            sim_weight = F.softmax(sim_weight / 0.1, dim=1)

        # counts for each class
        one_hot_label = torch.zeros(
            x.size(0) * self.k, self.classes, device=sim_labels.device
        )
        # [B*K, C]
        one_hot_label = one_hot_label.scatter(
            dim=-1, index=sim_labels.view(-1, 1), value=1.0
        )
        # weighted score ---> [B, C]
        pred_scores = torch.sum(
            one_hot_label.view(x.size(0), -1, self.classes)
            * sim_weight.unsqueeze(dim=-1),
            dim=1,
        )

        pred_scores = (pred_scores + 1e-5).clamp(max=1.0)
        return pred_scores


class MomentumQueue(nn.Module):
    def __init__(self, feature_dim, queue_size, temperature, k, classes, eps_ball=1.1):
        super(MomentumQueue, self).__init__()
        self.queue_size = queue_size
        self.index = 0
        self.temperature = temperature
        self.k = k
        self.classes = classes
        self.eps_ball = eps_ball

        # noinspection PyCallingNonCallable
        self.register_buffer("params", torch.tensor([-1]))
        stdv = 1.0 / math.sqrt(feature_dim / 3)
        memory = (
            torch.rand(self.queue_size, feature_dim, requires_grad=False)
            .mul_(2 * stdv)
            .add_(-stdv)
        )
        self.register_buffer("memory", memory)
        memory_label = torch.zeros(self.queue_size).long()
        self.register_buffer("memory_label", memory_label)

    def update_queue(self, k_all, k_label_all):
        with torch.no_grad():
            k_all = F.normalize(k_all)
            all_size = k_all.shape[0]
            out_ids = torch.fmod(
                torch.arange(all_size, dtype=torch.long).cuda() + self.index,
                self.queue_size,
            )
            self.memory.index_copy_(0, out_ids, k_all)
            self.memory_label.index_copy_(0, out_ids, k_label_all)
            self.index = (self.index + all_size) % self.queue_size

    def extend_test(self, k_all, k_label_all):
        with torch.no_grad():
            k_all = F.normalize(k_all)
            self.memory = torch.cat([self.memory, k_all], dim=0)
            self.memory_label = torch.cat([self.memory_label, k_label_all], dim=0)

    def forward(self, x, test=False):
        dist = torch.mm(
            F.normalize(x), self.memory.transpose(1, 0)
        )  # B * Q, memory already normalized
        max_batch_dist, _ = torch.max(dist, 1)
        # compute cos similarity between each feature vector and feature bank ---> [B, N]
        if self.eps_ball <= 1 and max_batch_dist.min() > self.eps_ball:
            sim_weight = torch.where(
                dist >= self.eps_ball,
                dist,
                torch.tensor(float("-inf")).float().to(x.device),
            )
            sim_labels = self.memory_label.expand(x.size(0), -1)
            sim_weight = F.softmax(sim_weight / self.temperature, dim=1)
            # counts for each class
            one_hot_label = torch.zeros(
                x.size(0) * self.memory_label.shape[0],
                self.classes,
                device=sim_labels.device,
            )
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(
                dim=-1, index=sim_labels.contiguous().view(-1, 1), value=1.0
            )
            # weighted score ---> [B, C]
        else:
            sim_weight, sim_indices = torch.topk(dist, k=self.k)
            sim_labels = torch.gather(
                self.memory_label.expand(x.size(0), -1), dim=-1, index=sim_indices
            )
            sim_weight = F.softmax(sim_weight / self.temperature, dim=1)

            one_hot_label = torch.zeros(
                x.size(0) * self.k, self.classes, device=sim_labels.device
            )
            one_hot_label = one_hot_label.scatter(
                dim=-1, index=sim_labels.view(-1, 1), value=1.0
            )
        # weighted score ---> [B, C]
        pred_scores = torch.sum(
            one_hot_label.view(x.size(0), -1, self.classes)
            * sim_weight.unsqueeze(dim=-1),
            dim=1,
        )

        pred_scores = (pred_scores + 1e-5).clamp(max=1.0)
        return pred_scores


class Model_with_Predictor(nn.Module):
    """
    Build a SimSiam model.
    """

    def __init__(self, args, base_encoder, dim=2048, pred_dim=512):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(Model_with_Predictor, self).__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder = base_encoder(num_classes=dim, zero_init_residual=True)

        # build a 3-layer projector
        prev_dim = self.encoder.fc.weight.shape[1]
        if args.mlp:
            self.encoder.fc = nn.Sequential(
                nn.Linear(prev_dim, prev_dim, bias=False),
                nn.BatchNorm1d(prev_dim),
                nn.ReLU(inplace=True),  # first layer
                nn.Linear(prev_dim, prev_dim, bias=False),
                nn.BatchNorm1d(prev_dim),
                nn.ReLU(inplace=True),  # second layer
                self.encoder.fc,
                nn.BatchNorm1d(dim, affine=False),
            )  # output layer
            self.encoder.fc[6].bias.requires_grad = (
                False  # hack: not use bias as it is followed by BN
            )

        # build a 2-layer predictor
        self.predictor = nn.Sequential(
            nn.Linear(dim, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True),  # hidden layer
            nn.Linear(pred_dim, dim),
        )  # output layer

    def forward(self, x):
        z = self.encoder(x)
        p = self.predictor(z)
        return p, z


def moment_update(model, model_ema, m):
    """model_ema = m * model_ema + (1 - m) model"""
    for p1, p2 in zip(model.module.encoder.parameters(), model_ema.parameters()):
        p2.data.mul_(m).add_(1 - m, p1.detach().data)


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.hparams = hparams

    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.

        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError


class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(
        self, featurizer, classifier, input_shape, num_classes, num_domains, hparams
    ):
        super(ERM, self).__init__(input_shape, num_classes, num_domains, hparams)
        # self.featurizer = networks.Featurizer(input_shape, self.hparams)
        # self.classifier = networks.Classifier(
        #     self.featurizer.n_outputs, num_classes, self.hparams["nonlinear_classifier"]
        # )

        self.featurizer = featurizer
        self.classifier = classifier

        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def predict(
        self,
        x1,
        x2,
        rssi1,
        rssi2,
    ):
        return self.network(
            x1,
            x2,
            rssi1,
            rssi2,
        )

    def forward(
        self,
        x1,
        x2,
        rssi1,
        rssi2,
    ):
        return self.predict(
            x1,
            x2,
            rssi1,
            rssi2,
        )


class KNN(Algorithm):
    """
    Empirical Risk Minimization with non-parametric classifier (ERMkNN)
    """

    def __init__(
        self,
        backbone,
        input_shape,
        num_classes,
        num_domains,
        hparams,
        device,
        scaler,
        bank,
    ):
        super(KNN, self).__init__(
            input_shape=input_shape,
            num_classes=num_classes,
            num_domains=num_domains,
            hparams=hparams,
        )

        self.device = device
        self.num_domains = num_domains
        # self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.featurizer = backbone
        self.scaler = scaler
        self.bank = bank
        self.num_classes = num_classes

        from domainbed.knn import MomentumQueue

        self.classifier = MomentumQueue(
            self.featurizer.backbone.num_features,
            self.hparams["queue_size"],
            self.hparams["temperature"],
            self.hparams["k"],
            num_classes,
            bank=bank,
        )
        self.eval_knn = None

        self.optimizer = torch.optim.AdamW(
            self.featurizer.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )
        from torch.optim.lr_scheduler import ReduceLROnPlateau

        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=500,
            cooldown=500,
            min_lr=1e-6,
            verbose=True,
        )
        self.max_epoch = 5000
        self.step = 0

    def update(self, minibatches, uda=None, merge=False):
        self.classifier.temperature = (
            -(1.0 - 0.1) / (1.0 * self.max_epoch) * self.step + 1.0
        )
        self.step += 1
        self.optimizer.zero_grad()
        if merge:
            all_x = [minibatch[0].to(self.device) for minibatch in minibatches]
            # print("all_x[0].shape", all_x[0].shape)
            # print("all_x.shape", len(all_x))
            all_y = minibatches[0][1].to(self.device)
            # print("all_y.shape", all_y.shape)
            z = self.featurizer(all_x[0], all_x[1])
            loss = F.nll_loss(torch.log(self.classifier(z)), all_y)
        else:
            all_x = minibatches[0].to(self.device)
            all_y = minibatches[1].to(self.device)
            z = self.featurizer(all_x)
            # loss = F.cross_entropy(self.classifier(z), all_y)
            loss = F.nll_loss(torch.log(self.classifier(z)), all_y)

        loss.backward()
        self.optimizer.step()

        # self.scaler.scale(loss).backward()
        # self.scaler.step(self.optimizer)
        # scale = self.scaler.get_scale()
        # self.scaler.update()
        # skip_lr_sched = scale > self.scaler.get_scale()

        # if not skip_lr_sched:
        #     self.scheduler.step(loss)

        self.classifier.update_queue(z, all_y)

        return {"loss": loss.item()}

    def predict(self, x, return_logits=False, merge=False):
        return self.predict_knn(x, return_logits, merge)

    def forward(self, x):
        return self.predict_knn(x)

    def predict_knn(self, x, return_logits=False, merge=False):
        if merge:
            z = self.featurizer(x[0], x[1])
        else:
            z = self.featurizer(x)
        if return_logits:
            return self.eval_knn(z, test=True), z
        else:
            return self.eval_knn(z, test=True)


class DRM(Algorithm):
    """
    Class-based Empirical Risk Minimization with Multi Classifier(ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(DRM, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.com = False
        self.gamma = 0.0
        self.lf_alg = hparams["lf_alg"]
        self.test_conf = 0.5
        self.num_class = num_classes
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        if self.lf_alg == "nn":
            self.discriminator = networks.MLP(
                self.featurizer.n_outputs, num_domains, self.hparams
            )
            self.disc_opt = torch.optim.Adam(
                list(self.discriminator.parameters()),
                lr=self.hparams["lr_d"],
                weight_decay=self.hparams["weight_decay_d"],
                betas=(self.hparams["beta1"], 0.9),
            )
        elif self.lf_alg == "cosine" or self.lf_alg[0] == "p":
            self.bsz = self.hparams["batch_size"]
            self.k = hparams["prototype_K"]
            self.ebd_domains = torch.zeros(
                (
                    num_domains,
                    hparams["prototype_K"] * self.bsz,
                    self.featurizer.n_outputs,
                )
            ).cuda()
            self.step = 0

        self.num_domains = num_domains
        self.classifier_list = nn.ModuleList(
            [
                networks.Classifier(
                    self.featurizer.n_outputs,
                    num_classes,
                    self.hparams["nonlinear_classifier"],
                )
                for i in range(self.num_domains + 1)
            ]
        )

        self.network = nn.Sequential(self.featurizer, self.classifier_list)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

    def update(self, minibatches, unlabeled=None):
        loss = torch.zeros(0)
        if self.lf_alg == "nn":
            logits_ls = []
        losses, y_hats = [], []
        for i, (x, y) in enumerate(minibatches):
            logits = self.featurizer(x)

            y_hat = self.classifier_list[i](logits)
            y_hat_all = self.classifier_list[-1](logits)

            if i == 0:
                loss = (F.cross_entropy(y_hat, y) + F.cross_entropy(y_hat_all, y)) / 2
            else:
                loss += (F.cross_entropy(y_hat, y) + F.cross_entropy(y_hat_all, y)) / 2
            losses.append(loss.item())
            if self.lf_alg == "nn":
                logits_ls.append(logits.detach())
            elif self.lf_alg == "cosine" or self.lf_alg[0] == "p":
                self.ebd_domains[
                    i, self.step * self.bsz : (self.step + 1) * self.bsz
                ] = logits.detach()

        if self.lf_alg == "cosine" or self.lf_alg[0] == "p":
            self.step = (self.step + 1) % self.k

        loss /= self.num_domains
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.lf_alg == "nn":
            device = "cuda" if minibatches[0][0].is_cuda else "cpu"
            disc_input = torch.cat(logits_ls, dim=0)
            disc_out = self.discriminator(disc_input)
            disc_labels = torch.cat(
                [
                    torch.full((x.shape[0],), i, dtype=torch.int64, device=device)
                    for i, (x, y) in enumerate(minibatches)
                ]
            )
            disc_loss = F.cross_entropy(disc_out, disc_labels)
            self.disc_opt.zero_grad()
            disc_loss.backward()
            self.disc_opt.step()
            return {"loss": loss.item()}

        return {"loss": loss.item()}

    def softmax_entropy(self, x: torch.Tensor):
        """Entropy of softmax distribution from logits."""
        return -(x.softmax(1) * x.log_softmax(1)).sum(1)

    def forward(self, x):
        logits = self.featurizer(x)
        entropy = torch.tensor(1e10)
        result = None
        ents, y_hats = [], []
        for i in range(self.num_domains + 1):
            y_hat = self.classifier_list[i](logits)
            ent = self.softmax_entropy(y_hat).mean()
            ents.append(ent.item())
            if ent < entropy:
                entropy = ent
                result = y_hat
        return result

    def nn_predict(self, x):
        logits = self.featurizer(x)
        disc = self.discriminator(logits)
        idx = disc.argmax()

        if not self.com:
            return self.classifier_list[idx](logits), None
        ents = disc.detach().cpu()
        y_hats = []

        for i in range(self.num_domains):
            y_hat = self.classifier_list[i](logits)
            y_hats.append(F.softmax(y_hat))

        weight = F.softmax(disc).squeeze()
        com_result = torch.zeros_like(y_hats[0])
        for i in range(self.num_domains):
            com_result += weight[i] * y_hats[i]
        return com_result

    def cosine_predict(self, x):
        logits = self.featurizer(x)
        logits_domains = self.ebd_domains.mean(dim=-2)
        logits_sim = logits.repeat(logits_domains.shape[0], 1)

        if self.lf_alg == "cosine":
            sims = F.cosine_similarity(logits_sim, logits_domains)
        elif self.lf_alg[0] == "p":
            p = int(self.lf_alg[1])
            logits_sim = logits_sim - logits_domains
            sims = torch.norm(logits_sim, p=p, dim=1)

        idx = sims.argmin()

        if not self.com:
            return self.classifier_list[idx](logits)

        ents = sims.detach().cpu()
        y_hats = []

        for i in range(self.num_domains):
            y_hat = self.classifier_list[i](logits)
            y_hats.append(torch.nn.functional.softmax(y_hat, dim=1))

        if self.lf_alg == "cosine":
            weight = sims
            weight /= torch.sum(weight)
        elif self.lf_alg[0] == "p":
            weight = 1.0 / (np.array(ents) ** self.gamma)
            weight /= np.sum(weight)
        print(weight)
        com_result = torch.zeros_like(y_hats[0])
        for i in range(self.num_domains):
            com_result += weight[i] * y_hats[i]
        return com_result

    def predict(self, x):
        if self.lf_alg == "entropy":
            return self(x)
        elif self.lf_alg == "nn":
            return self.nn_predict(x)
        elif self.lf_alg == "cosine" or self.lf_alg[0] == "p":
            return self.cosine_predict(x)
        else:
            raise NotImplementedError


class ARM(ERM):
    """Adaptive Risk Minimization (ARM)"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        original_input_shape = input_shape
        input_shape = (1 + original_input_shape[0],) + original_input_shape[1:]
        super(ARM, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.context_net = networks.ContextNet(original_input_shape)
        self.support_size = hparams["batch_size"]

    def predict(self, x):
        batch_size, c, h, w = x.shape
        if batch_size % self.support_size == 0:
            meta_batch_size = batch_size // self.support_size
            support_size = self.support_size
        else:
            meta_batch_size, support_size = 1, batch_size
        context = self.context_net(x)
        context = context.reshape((meta_batch_size, support_size, 1, h, w))
        context = context.mean(dim=1)
        context = torch.repeat_interleave(context, repeats=support_size, dim=0)
        x = torch.cat([x, context], dim=1)
        return self.network(x)


class AbstractDANN(Algorithm):
    """Domain-Adversarial Neural Networks (abstract class)"""

    def __init__(
        self, input_shape, num_classes, num_domains, hparams, conditional, class_balance
    ):

        super(AbstractDANN, self).__init__(
            input_shape, num_classes, num_domains, hparams
        )

        self.register_buffer("update_count", torch.tensor([0]))
        self.conditional = conditional
        self.class_balance = class_balance

        # Algorithms
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs, num_classes, self.hparams["nonlinear_classifier"]
        )
        self.discriminator = networks.MLP(
            self.featurizer.n_outputs, num_domains, self.hparams
        )
        self.class_embeddings = nn.Embedding(num_classes, self.featurizer.n_outputs)

        # Optimizers
        self.disc_opt = torch.optim.Adam(
            (
                list(self.discriminator.parameters())
                + list(self.class_embeddings.parameters())
            ),
            lr=self.hparams["lr_d"],
            weight_decay=self.hparams["weight_decay_d"],
            betas=(self.hparams["beta1"], 0.9),
        )

        self.gen_opt = torch.optim.Adam(
            (list(self.featurizer.parameters()) + list(self.classifier.parameters())),
            lr=self.hparams["lr_g"],
            weight_decay=self.hparams["weight_decay_g"],
            betas=(self.hparams["beta1"], 0.9),
        )

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        self.update_count += 1
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        all_z = self.featurizer(all_x)
        if self.conditional:
            disc_input = all_z + self.class_embeddings(all_y)
        else:
            disc_input = all_z
        disc_out = self.discriminator(disc_input)
        disc_labels = torch.cat(
            [
                torch.full((x.shape[0],), i, dtype=torch.int64, device=device)
                for i, (x, y) in enumerate(minibatches)
            ]
        )

        if self.class_balance:
            y_counts = F.one_hot(all_y).sum(dim=0)
            weights = 1.0 / (y_counts[all_y] * y_counts.shape[0]).float()
            disc_loss = F.cross_entropy(disc_out, disc_labels, reduction="none")
            disc_loss = (weights * disc_loss).sum()
        else:
            disc_loss = F.cross_entropy(disc_out, disc_labels)

        disc_softmax = F.softmax(disc_out, dim=1)
        input_grad = autograd.grad(
            disc_softmax[:, disc_labels].sum(), [disc_input], create_graph=True
        )[0]
        grad_penalty = (input_grad**2).sum(dim=1).mean(dim=0)
        disc_loss += self.hparams["grad_penalty"] * grad_penalty

        d_steps_per_g = self.hparams["d_steps_per_g_step"]
        if self.update_count.item() % (1 + d_steps_per_g) < d_steps_per_g:

            self.disc_opt.zero_grad()
            disc_loss.backward()
            self.disc_opt.step()
            return {"disc_loss": disc_loss.item()}
        else:
            all_preds = self.classifier(all_z)
            classifier_loss = F.cross_entropy(all_preds, all_y)
            gen_loss = classifier_loss + (self.hparams["lambda"] * -disc_loss)
            self.disc_opt.zero_grad()
            self.gen_opt.zero_grad()
            gen_loss.backward()
            self.gen_opt.step()
            return {"gen_loss": gen_loss.item()}

    def predict(self, x):
        return self.classifier(self.featurizer(x))


class DANN(AbstractDANN):
    """Unconditional DANN"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(DANN, self).__init__(
            input_shape,
            num_classes,
            num_domains,
            hparams,
            conditional=False,
            class_balance=False,
        )


class CDANN(AbstractDANN):
    """Conditional DANN"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CDANN, self).__init__(
            input_shape,
            num_classes,
            num_domains,
            hparams,
            conditional=True,
            class_balance=True,
        )


class IRM(ERM):
    """Invariant Risk Minimization"""

    def __init__(
        self, featurizer, classifier, input_shape, num_classes, num_domains, hparams
    ):
        super(IRM, self).__init__(
            featurizer, classifier, input_shape, num_classes, num_domains, hparams
        )
        self.register_buffer("update_count", torch.tensor([0]))

    @staticmethod
    def _irm_penalty(logits, y):
        device = "cuda" if logits[0][0].is_cuda else "cpu"
        # print("logits[::2]", logits[::2])
        # print("logits[1::2]", logits[1::2])
        # print("y[::2]", y[::2])
        # print("y[1::2]", y[1::2])
        scale = torch.tensor(1.0).to(device).requires_grad_()
        loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
        loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        return result

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        penalty_weight = (
            self.hparams["irm_lambda"]
            if self.update_count >= self.hparams["irm_penalty_anneal_iters"]
            else 1.0
        )
        nll = 0.0
        penalty = 0.0

        all_x = torch.cat([x for x, y in minibatches])
        all_logits = self.network(all_x)
        all_logits_idx = 0
        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx : all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll += F.cross_entropy(logits, y)
            penalty += self._irm_penalty(logits, y)
        nll /= len(minibatches)
        penalty /= len(minibatches)
        loss = nll + (penalty_weight * penalty)

        if self.update_count == self.hparams["irm_penalty_anneal_iters"]:
            # Reset Adam, because it doesn't like the sharp jump in gradient
            # magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams["weight_decay"],
            )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {"loss": loss.item(), "nll": nll.item(), "penalty": penalty.item()}


class IBIRM(IRM):
    """IB Invariant Risk Minimization"""

    def __init__(
        self,
        featurizer,
        classifier,
        scaler,
        input_shape,
        num_classes,
        num_domains,
        hparams,
        device,
    ):
        super(IBIRM, self).__init__(
            featurizer, classifier, input_shape, num_classes, num_domains, hparams
        )

        self.scaler = scaler
        self.device = device

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=500,
            cooldown=500,
            min_lr=1e-6,
            verbose=True,
        )

    def update(self, minibatches, unlabeled=None):
        penalty_weight = (
            self.hparams["irm_lambda"]
            if self.update_count >= self.hparams["irm_penalty_anneal_iters"]
            else 0.0
        )  # todo

        ib_penalty_weight = (
            self.hparams["ib_lambda"]
            if self.update_count >= self.hparams["ib_penalty_anneal_iters"]
            else 0.0
        )

        nll = 0.0
        penalty = 0.0

        all_head = {
            "x1": minibatches[0][0],
            "y1": minibatches[0][1],
            "rssi1": minibatches[0][2],
        }
        all_end = {
            "x2": minibatches[1][0],
            "y2": minibatches[1][1],
            "rssi2": minibatches[1][2],
        }

        # print("all_head x1", all_head["x1"].shape)
        # print("all_head rssi1", all_head["rssi1"].shape)
        # print("all_head y1", all_head["y1"].shape)

        # all_x = torch.cat([x for x, y in minibatches])
        # all_y = torch.cat([y for x, y in minibatches])
        with autocast():
            inter_logits = self.featurizer(
                {
                    "x1": all_head["x1"],
                    "x2": all_end["x2"],
                    "rssi1": all_head["rssi1"],
                    "rssi2": all_end["rssi2"],
                }
            )
            all_logits = self.classifier(inter_logits)
            all_logits_idx = 0

            logits = all_logits[
                all_logits_idx : all_logits_idx + all_head["x1"].shape[0]
            ]
            # print("logits", logits)
            # print("logits.shape", logits.shape)
            all_logits_idx += all_head["x1"].shape[0]
            nll += F.cross_entropy(logits, all_head["y1"])
            penalty += self._irm_penalty(logits, all_head["y1"])
            nll /= len(minibatches[0])
            penalty /= len(minibatches[0])
            loss = nll + (penalty_weight * penalty)
            if penalty_weight > 1:
                loss /= 1 + penalty_weight

            var_loss = inter_logits.var(dim=0).mean()
            loss += ib_penalty_weight * var_loss

        if self.update_count == self.hparams["irm_penalty_anneal_iters"]:
            # Reset Adam, because it doesn't like the sharp jump in gradient
            # magnitudes that happens at this step.
            self.optimizer = torch.optim.AdamW(
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams["weight_decay"],
            )

            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.5,
                patience=250,
                cooldown=250,
                min_lr=1e-6,
                verbose=True,
            )

        self.scaler.scale(loss).backward()

        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 10)

        self.scaler.step(self.optimizer)
        scale = self.scaler.get_scale()
        self.scaler.update()
        skip_lr_sched = scale > self.scaler.get_scale()

        if not skip_lr_sched:
            self.scheduler.step(loss)

        self.update_count += 1
        return {
            "loss": loss.item(),
            "nll": nll.item(),
            "penalty": penalty.item(),
            "var": var_loss.item(),
        }

    def predict(self, x):
        return self.classifier(self.featurizer(x))


class VREx(ERM):
    """V-REx algorithm from http://arxiv.org/abs/2003.00688"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(VREx, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.register_buffer("update_count", torch.tensor([0]))

    def update(self, minibatches, unlabeled=None):
        if self.update_count >= self.hparams["vrex_penalty_anneal_iters"]:
            penalty_weight = self.hparams["vrex_lambda"]
        else:
            penalty_weight = 1.0

        nll = 0.0

        all_x = torch.cat([x for x, y in minibatches])
        all_logits = self.network(all_x)
        all_logits_idx = 0
        losses = torch.zeros(len(minibatches))
        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx : all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll = F.cross_entropy(logits, y)
            losses[i] = nll

        mean = losses.mean()
        penalty = ((losses - mean) ** 2).mean()
        loss = mean + penalty_weight * penalty

        if self.update_count == self.hparams["vrex_penalty_anneal_iters"]:
            # Reset Adam (like IRM), because it doesn't like the sharp jump in
            # gradient magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams["weight_decay"],
            )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {"loss": loss.item(), "nll": nll.item(), "penalty": penalty.item()}


class Mixup(ERM):
    """
    Mixup of minibatches from different domains
    https://arxiv.org/pdf/2001.00677.pdf
    https://arxiv.org/pdf/1912.01805.pdf
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Mixup, self).__init__(input_shape, num_classes, num_domains, hparams)

    def update(self, minibatches, unlabeled=None):
        objective = 0

        for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
            lam = np.random.beta(
                self.hparams["mixup_alpha"], self.hparams["mixup_alpha"]
            )

            x = lam * xi + (1 - lam) * xj
            predictions = self.predict(x)

            objective += lam * F.cross_entropy(predictions, yi)
            objective += (1 - lam) * F.cross_entropy(predictions, yj)

        objective /= len(minibatches)

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {"loss": objective.item()}


class GroupDRO(ERM):
    """
    Robust ERM minimizes the error at the worst minibatch
    Algorithm 1 from [https://arxiv.org/pdf/1911.08731.pdf]
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(GroupDRO, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.register_buffer("q", torch.Tensor())

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"

        if not len(self.q):
            self.q = torch.ones(len(minibatches)).to(device)

        losses = torch.zeros(len(minibatches)).to(device)

        for m in range(len(minibatches)):
            x, y = minibatches[m]
            losses[m] = F.cross_entropy(self.predict(x), y)
            self.q[m] *= (self.hparams["groupdro_eta"] * losses[m].data).exp()

        self.q /= self.q.sum()

        loss = torch.dot(losses, self.q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}


class MLDG(ERM):
    """
    Model-Agnostic Meta-Learning
    Algorithm 1 / Equation (3) from: https://arxiv.org/pdf/1710.03463.pdf
    Related: https://arxiv.org/pdf/1703.03400.pdf
    Related: https://arxiv.org/pdf/1910.13580.pdf
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MLDG, self).__init__(input_shape, num_classes, num_domains, hparams)

    def update(self, minibatches, unlabeled=None):
        """
        Terms being computed:
            * Li = Loss(xi, yi, params)
            * Gi = Grad(Li, params)

            * Lj = Loss(xj, yj, Optimizer(params, grad(Li, params)))
            * Gj = Grad(Lj, params)

            * params = Optimizer(params, Grad(Li + beta * Lj, params))
            *        = Optimizer(params, Gi + beta * Gj)

        That is, when calling .step(), we want grads to be Gi + beta * Gj

        For computational efficiency, we do not compute second derivatives.
        """
        num_mb = len(minibatches)
        objective = 0

        self.optimizer.zero_grad()
        for p in self.network.parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p)

        for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
            # fine tune clone-network on task "i"
            inner_net = copy.deepcopy(self.network)

            inner_opt = torch.optim.Adam(
                inner_net.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams["weight_decay"],
            )

            inner_obj = F.cross_entropy(inner_net(xi), yi)

            inner_opt.zero_grad()
            inner_obj.backward()
            inner_opt.step()

            # The network has now accumulated gradients Gi
            # The clone-network has now parameters P - lr * Gi
            for p_tgt, p_src in zip(self.network.parameters(), inner_net.parameters()):
                if p_src.grad is not None:
                    p_tgt.grad.data.add_(p_src.grad.data / num_mb)

            # `objective` is populated for reporting purposes
            objective += inner_obj.item()

            # this computes Gj on the clone-network
            loss_inner_j = F.cross_entropy(inner_net(xj), yj)
            grad_inner_j = autograd.grad(
                loss_inner_j, inner_net.parameters(), allow_unused=True
            )

            # `objective` is populated for reporting purposes
            objective += (self.hparams["mldg_beta"] * loss_inner_j).item()

            for p, g_j in zip(self.network.parameters(), grad_inner_j):
                if g_j is not None:
                    p.grad.data.add_(self.hparams["mldg_beta"] * g_j.data / num_mb)

            # The network has now accumulated gradients Gi + beta * Gj
            # Repeat for all train-test splits, do .step()

        objective /= len(minibatches)

        self.optimizer.step()

        return {"loss": objective}

    # This commented "update" method back-propagates through the gradients of
    # the inner update, as suggested in the original MAML paper.  However, this
    # is twice as expensive as the uncommented "update" method, which does not
    # compute second-order derivatives, implementing the First-Order MAML
    # method (FOMAML) described in the original MAML paper.

    # def update(self, minibatches, unlabeled=None):
    #     objective = 0
    #     beta = self.hparams["beta"]
    #     inner_iterations = self.hparams["inner_iterations"]

    #     self.optimizer.zero_grad()

    #     with higher.innerloop_ctx(self.network, self.optimizer,
    #         copy_initial_weights=False) as (inner_network, inner_optimizer):

    #         for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
    #             for inner_iteration in range(inner_iterations):
    #                 li = F.cross_entropy(inner_network(xi), yi)
    #                 inner_optimizer.step(li)
    #
    #             objective += F.cross_entropy(self.network(xi), yi)
    #             objective += beta * F.cross_entropy(inner_network(xj), yj)

    #         objective /= len(minibatches)
    #         objective.backward()
    #
    #     self.optimizer.step()
    #
    #     return objective


class AbstractMMD(ERM):
    """
    Perform ERM while matching the pair-wise domain feature distributions
    using MMD (abstract class)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams, gaussian):
        super(AbstractMMD, self).__init__(
            input_shape, num_classes, num_domains, hparams
        )
        if gaussian:
            self.kernel_type = "gaussian"
        else:
            self.kernel_type = "mean_cov"

    def my_cdist(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(
            x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2
        ).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100, 1000]):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def mmd(self, x, y):
        if self.kernel_type == "gaussian":
            Kxx = self.gaussian_kernel(x, x).mean()
            Kyy = self.gaussian_kernel(y, y).mean()
            Kxy = self.gaussian_kernel(x, y).mean()
            return Kxx + Kyy - 2 * Kxy
        else:
            mean_x = x.mean(0, keepdim=True)
            mean_y = y.mean(0, keepdim=True)
            cent_x = x - mean_x
            cent_y = y - mean_y
            cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
            cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

            mean_diff = (mean_x - mean_y).pow(2).mean()
            cova_diff = (cova_x - cova_y).pow(2).mean()

            return mean_diff + cova_diff

    def update(self, minibatches, unlabeled=None):
        objective = 0
        penalty = 0
        nmb = len(minibatches)

        features = [self.featurizer(xi) for xi, _ in minibatches]
        classifs = [self.classifier(fi) for fi in features]
        targets = [yi for _, yi in minibatches]

        for i in range(nmb):
            objective += F.cross_entropy(classifs[i], targets[i])
            for j in range(i + 1, nmb):
                penalty += self.mmd(features[i], features[j])

        objective /= nmb
        if nmb > 1:
            penalty /= nmb * (nmb - 1) / 2

        self.optimizer.zero_grad()
        (objective + (self.hparams["mmd_gamma"] * penalty)).backward()
        self.optimizer.step()

        if torch.is_tensor(penalty):
            penalty = penalty.item()

        return {"loss": objective.item(), "penalty": penalty}


class MMD(AbstractMMD):
    """
    MMD using Gaussian kernel
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MMD, self).__init__(
            input_shape, num_classes, num_domains, hparams, gaussian=True
        )


class CORAL(AbstractMMD):
    """
    MMD using mean and covariance difference
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CORAL, self).__init__(
            input_shape, num_classes, num_domains, hparams, gaussian=False
        )


class MTL(Algorithm):
    """
    A neural network version of
    Domain Generalization by Marginal Transfer Learning
    (https://arxiv.org/abs/1711.07910)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MTL, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs * 2,
            num_classes,
            self.hparams["nonlinear_classifier"],
        )
        self.optimizer = torch.optim.Adam(
            list(self.featurizer.parameters()) + list(self.classifier.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

        self.register_buffer(
            "embeddings", torch.zeros(num_domains, self.featurizer.n_outputs)
        )

        self.ema = self.hparams["mtl_ema"]

    def update(self, minibatches, unlabeled=None):
        loss = 0
        for env, (x, y) in enumerate(minibatches):
            loss += F.cross_entropy(self.predict(x, env), y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def update_embeddings_(self, features, env=None):
        return_embedding = features.mean(0)

        if env is not None:
            return_embedding = (
                self.ema * return_embedding + (1 - self.ema) * self.embeddings[env]
            )

            self.embeddings[env] = return_embedding.clone().detach()

        return return_embedding.view(1, -1).repeat(len(features), 1)

    def predict(self, x, env=None):
        features = self.featurizer(x)
        embedding = self.update_embeddings_(features, env).normal_()
        return self.classifier(torch.cat((features, embedding), 1))


class SagNet(Algorithm):
    """
    Style Agnostic Network
    Algorithm 1 from: https://arxiv.org/abs/1910.11645
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(SagNet, self).__init__(input_shape, num_classes, num_domains, hparams)
        # featurizer network
        self.network_f = networks.Featurizer(input_shape, self.hparams)
        # content network
        self.network_c = networks.Classifier(
            self.network_f.n_outputs, num_classes, self.hparams["nonlinear_classifier"]
        )
        # style network
        self.network_s = networks.Classifier(
            self.network_f.n_outputs, num_classes, self.hparams["nonlinear_classifier"]
        )

        # # This commented block of code implements something closer to the
        # # original paper, but is specific to ResNet and puts in disadvantage
        # # the other algorithms.
        # resnet_c = networks.Featurizer(input_shape, self.hparams)
        # resnet_s = networks.Featurizer(input_shape, self.hparams)
        # # featurizer network
        # self.network_f = torch.nn.Sequential(
        #         resnet_c.network.conv1,
        #         resnet_c.network.bn1,
        #         resnet_c.network.relu,
        #         resnet_c.network.maxpool,
        #         resnet_c.network.layer1,
        #         resnet_c.network.layer2,
        #         resnet_c.network.layer3)
        # # content network
        # self.network_c = torch.nn.Sequential(
        #         resnet_c.network.layer4,
        #         resnet_c.network.avgpool,
        #         networks.Flatten(),
        #         resnet_c.network.fc)
        # # style network
        # self.network_s = torch.nn.Sequential(
        #         resnet_s.network.layer4,
        #         resnet_s.network.avgpool,
        #         networks.Flatten(),
        #         resnet_s.network.fc)

        def opt(p):
            return torch.optim.Adam(
                p, lr=hparams["lr"], weight_decay=hparams["weight_decay"]
            )

        self.optimizer_f = opt(self.network_f.parameters())
        self.optimizer_c = opt(self.network_c.parameters())
        self.optimizer_s = opt(self.network_s.parameters())
        self.weight_adv = hparams["sag_w_adv"]

    def forward_c(self, x):
        # learning content network on randomized style
        return self.network_c(self.randomize(self.network_f(x), "style"))

    def forward_s(self, x):
        # learning style network on randomized content
        return self.network_s(self.randomize(self.network_f(x), "content"))

    def randomize(self, x, what="style", eps=1e-5):
        device = "cuda" if x.is_cuda else "cpu"
        sizes = x.size()
        alpha = torch.rand(sizes[0], 1).to(device)

        if len(sizes) == 4:
            x = x.view(sizes[0], sizes[1], -1)
            alpha = alpha.unsqueeze(-1)

        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        x = (x - mean) / (var + eps).sqrt()

        idx_swap = torch.randperm(sizes[0])
        if what == "style":
            mean = alpha * mean + (1 - alpha) * mean[idx_swap]
            var = alpha * var + (1 - alpha) * var[idx_swap]
        else:
            x = x[idx_swap].detach()

        x = x * (var + eps).sqrt() + mean
        return x.view(*sizes)

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])

        # learn content
        self.optimizer_f.zero_grad()
        self.optimizer_c.zero_grad()
        loss_c = F.cross_entropy(self.forward_c(all_x), all_y)
        loss_c.backward()
        self.optimizer_f.step()
        self.optimizer_c.step()

        # learn style
        self.optimizer_s.zero_grad()
        loss_s = F.cross_entropy(self.forward_s(all_x), all_y)
        loss_s.backward()
        self.optimizer_s.step()

        # learn adversary
        self.optimizer_f.zero_grad()
        loss_adv = -F.log_softmax(self.forward_s(all_x), dim=1).mean(1).mean()
        loss_adv = loss_adv * self.weight_adv
        loss_adv.backward()
        self.optimizer_f.step()

        return {
            "loss_c": loss_c.item(),
            "loss_s": loss_s.item(),
            "loss_adv": loss_adv.item(),
        }

    def predict(self, x):
        return self.network_c(self.network_f(x))


class RSC(ERM):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(RSC, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.drop_f = (1 - hparams["rsc_f_drop_factor"]) * 100
        self.drop_b = (1 - hparams["rsc_b_drop_factor"]) * 100
        self.num_classes = num_classes

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"

        # inputs
        all_x = torch.cat([x for x, y in minibatches])
        # labels
        all_y = torch.cat([y for _, y in minibatches])
        # one-hot labels
        all_o = torch.nn.functional.one_hot(all_y, self.num_classes)
        # features
        all_f = self.featurizer(all_x)
        # predictions
        all_p = self.classifier(all_f)

        # Equation (1): compute gradients with respect to representation
        all_g = autograd.grad((all_p * all_o).sum(), all_f)[0]

        # Equation (2): compute top-gradient-percentile mask
        percentiles = np.percentile(all_g.cpu(), self.drop_f, axis=1)
        percentiles = torch.Tensor(percentiles)
        percentiles = percentiles.unsqueeze(1).repeat(1, all_g.size(1))
        mask_f = all_g.lt(percentiles.to(device)).float()

        # Equation (3): mute top-gradient-percentile activations
        all_f_muted = all_f * mask_f

        # Equation (4): compute muted predictions
        all_p_muted = self.classifier(all_f_muted)

        # Section 3.3: Batch Percentage
        all_s = F.softmax(all_p, dim=1)
        all_s_muted = F.softmax(all_p_muted, dim=1)
        changes = (all_s * all_o).sum(1) - (all_s_muted * all_o).sum(1)
        percentile = np.percentile(changes.detach().cpu(), self.drop_b)
        mask_b = changes.lt(percentile).float().view(-1, 1)
        mask = torch.logical_or(mask_f, mask_b).float()

        # Equations (3) and (4) again, this time mutting over examples
        all_p_muted_again = self.classifier(all_f * mask)

        # Equation (5): update
        loss = F.cross_entropy(all_p_muted_again, all_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}


class SD(ERM):
    """
    Gradient Starvation: A Learning Proclivity in Neural Networks
    Equation 25 from [https://arxiv.org/pdf/2011.09468.pdf]
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(SD, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.sd_reg = hparams["sd_reg"]

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        all_p = self.predict(all_x)

        loss = F.cross_entropy(all_p, all_y)
        penalty = (all_p**2).mean()
        objective = loss + self.sd_reg * penalty

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {"loss": loss.item(), "penalty": penalty.item()}


def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = copy.deepcopy(model.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=False)
    optimizer.load_state_dict(optimizer_state)


class RTCSI:
    def __init__(self, x):
        self.x = x

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx]


class CSIDataset:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class AdaNPC(Algorithm):
    def __init__(self, input_shape, num_classes, num_domains, hparams, algorithm, bank):
        """
        Hparams
        -------
        alpha (float) : learning rate coefficient
        beta (float) : threshold
        gamma (int) : number of updates
        """
        super().__init__(input_shape, num_classes, num_domains, hparams)
        # from domainbed.knn import MomentumQueue

        self.beta = hparams["beta"]
        self.model = algorithm
        self.bank = bank

        # self.classifier = MomentumQueue(
        #     self.model.featurizer.embedding,
        #     1,
        #     temperature=hparams["temperature"],
        #     k=self.hparams["k"],
        #     classes=num_classes,
        # )

    def forward(self, x, adapt=False):
        if adapt:
            outputs = self.forward_and_adapt(x)
        else:
            outputs = self.model.classifier(x)

        return outputs

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x):
        """Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        # forward
        p = self.model.classifier(x)
        confidences, predict = p.softmax(1).max(1)
        predict = predict[confidences >= self.beta]
        if predict.shape[0] > 0:
            if self.bank == "memory":
                self.model.classifier.extend_test(x[confidences >= self.beta], predict)
            else:
                self.model.classifier.ema_update(x[confidences >= self.beta], predict)
        return p

    def predict(self, x, adapt=False):
        return self(x, adapt)

    def reset(self, reset_zero=False):
        if reset_zero:
            self.classifier.memory = self.classifier.memory[:0, :]
            self.classifier.memory_label = self.classifier.memory_label[:0]
        else:
            self.classifier.memory = self.classifier.memory[
                : self.classifier.queue_size, :
            ]
            self.classifier.memory_label = self.classifier.memory_label[
                : self.classifier.queue_size
            ]

    def reset_params(self, hparams):
        self.beta = hparams["beta"]
        self.classifier.k = hparams["k"]
        self.classifier.temperature = hparams["temperature"]
        self.reset()


class AdaNPCBN(AdaNPC):
    def __init__(self, input_shape, num_classes, num_domains, hparams, algorithm):
        """
        Hparams
        -------
        alpha (float) : learning rate coefficient
        beta (float) : threshold
        gamma (int) : number of updates
        """
        super().__init__(input_shape, num_classes, num_domains, hparams, algorithm)
        from domainbed.knn import MomentumQueue

        self.beta = 0.1
        self.model = algorithm
        self.bn, self.optimizer = self.configure_model_optimizer(algorithm, alpha=0.01)
        self.steps = 3
        self.classifier = MomentumQueue(
            self.model.featurizer.embedding,
            1,
            temperature=0.01,
            k=self.hparams["k"],
            classes=num_classes,
            eps_ball=1.1,
        )
        self.model_state, self.optimizer_state = copy_model_and_optimizer(
            self.model, self.optimizer
        )

    def reset_params(self, hparams):
        self.beta = hparams["beta"]
        self.classifier.k = hparams["k"]
        self.classifier.temperature = hparams["temperature"]
        self.steps = hparams["gamma"]
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(
            self.model, self.optimizer, self.model_state, self.optimizer_state
        )
        self.bn, self.optimizer = self.configure_model_optimizer(
            self.model, alpha=hparams["alpha"]
        )
        self.reset()

    def forward(self, x, adapt=False):
        if adapt:
            for _ in range(self.steps):
                outputs = self.forward_and_adapt(x)
        else:
            outputs = self.model.classifier(x)
        return outputs

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x):
        """Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        # forward
        x = self.bn(x)
        p = self.classifier(x)
        confidences, predict = p.softmax(1).max(1)
        predict = predict[confidences >= self.beta]
        if predict.shape[0] > 0:
            self.classifier.extend_test(x[confidences >= self.beta], predict)
        self.optimizer.zero_grad()
        loss = softmax_entropy(p).mean(0)
        loss.backward()
        self.optimizer.step()
        return p

    def configure_model_optimizer(self, algorithm, alpha):
        bn = nn.BatchNorm1d(algorithm.featurizer.embedding).cuda()
        optimizer = torch.optim.AdamW(
            bn.parameters(),
            lr=algorithm.hparams["lr"] * alpha,
            weight_decay=algorithm.hparams["weight_decay"],
        )
        # adapted_algorithm.classifier.predict = lambda self, x: self(x)
        return bn, optimizer


def generate_featurelized_loader(loader, network, classifier, device, batch_size=64):
    """
    The classifier adaptation does not need to repeat the heavy forward path,
    We speeded up the experiments by converting the observations into representations.
    """
    z_list = []
    y_list = []
    p_list = []
    network.eval()
    classifier.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            z = network(x)
            p = classifier(z)

            z_list.append(z.detach().cpu())
            y_list.append(y.detach().cpu())
            p_list.append(p.detach().cpu())
            # p_list.append(p.argmax(1).float().cpu().detach())
    network.train()
    classifier.train()
    z = torch.cat(z_list)
    y = torch.cat(y_list)
    p = torch.cat(p_list)
    ent = softmax_entropy(p)
    py = p.argmax(1).float().cpu().detach()
    dataset1, dataset2 = CSIDataset(z, y), CSIDataset(z, py)
    loader1 = torch.utils.data.DataLoader(
        dataset1, batch_size=batch_size, shuffle=False, drop_last=False
    )
    loader2 = torch.utils.data.DataLoader(
        dataset2, batch_size=batch_size, shuffle=False, drop_last=False
    )
    return loader1, loader2, ent, z, y
