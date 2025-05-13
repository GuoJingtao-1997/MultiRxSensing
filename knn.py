import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MomentumQueue(nn.Module):
    def __init__(
        self,
        feature_dim,
        queue_size,
        temperature,
        k,
        classes,
        eps_ball=1.1,
        alpha=0.9,
        bank="memory",
    ):
        super(MomentumQueue, self).__init__()
        self.queue_size = queue_size
        self.index = 0
        self.temperature = temperature
        self.k = k
        self.classes = classes
        self.eps_ball = eps_ball
        self.alpha = alpha
        self.bank = bank

        # Ensure total memory isn't exceeded
        self.max_allowed_samples = self.queue_size // self.classes

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

        anchor_memory = torch.zeros(classes, memory.shape[1])
        self.register_buffer("anchor_memory", anchor_memory)
        anchor_memory_label = torch.zeros(classes).long()
        self.register_buffer("anchor_memory_label", anchor_memory_label)

        self.class_indices = {}
        self.class_positions = {}

    def _extend_single(self, feature, label):
        """
        Extends the memory bank with a single feature and label.
        Warning: This makes the queue size dynamic and might invalidate
        pre-calculated sizes or assumptions based on a fixed queue_size.
        """
        with torch.no_grad():
            # Ensure feature and label are on the correct device and have the right shape
            feature = feature.to(self.memory.device).unsqueeze(0)
            label_tensor = torch.tensor(
                [label], device=self.memory_label.device, dtype=self.memory_label.dtype
            )

            # Concatenate the new data
            self.memory = torch.cat([self.memory, feature], dim=0)
            self.memory_label = torch.cat([self.memory_label, label_tensor], dim=0)

            # Update queue_size attribute to reflect the new size
            new_index = self.memory.shape[0] - 1
            self.queue_size = self.memory.shape[0]

            # Update tracking dictionaries for the new entry
            if label not in self.class_positions:
                self.class_positions[label] = []
                self.class_indices[label] = 0  # Initialize FIFO index for the class
            self.class_positions[label].append(new_index)

            print(
                f"Extended memory bank. New size: {self.queue_size}. Added label {label} at index {new_index}"
            )

    def sort_dist_by_label(self, dist, memory_label):
        """
        Sort distance matrix by label ranges and return sorted similarities and indices.

        Args:
            dist: Tensor of shape [B, N] containing similarity/distance values
            memory_label: Tensor of shape [N] containing label information for memory bank

        Returns:
            sorted_similarities: Dictionary containing sorted similarities for each class
            sorted_indices: Dictionary containing corresponding sorted indices for each class
        """
        sorted_similarities = {}
        sorted_indices = {}

        with torch.no_grad():
            for class_idx in range(self.classes):
                # Get indices for current class in memory bank
                class_mask = memory_label == class_idx

                # Select distances for current class
                class_dist = dist[:, class_mask]

                # Sort similarities for current class
                similarities, indices = torch.sort(class_dist, dim=1)

                print(
                    f"similarities {similarities}, similarities.shape {similarities.shape}"
                )
                print(f"indices {indices}, indices.shape {indices.shape}")

                # Convert indices to original memory indices
                memory_indices = torch.where(class_mask)[0]
                actual_indices = memory_indices[indices[:, 0]]

                print(
                    f"memory_indices {memory_indices}, memory_indices.shape {memory_indices.shape}"
                )
                print(
                    f"actual_indices {actual_indices}, actual_indices.shape {actual_indices.shape}"
                )

                # Store results for current class
                sorted_similarities[class_idx] = similarities
                sorted_indices[class_idx] = actual_indices

        return sorted_similarities, sorted_indices

    def update_class_queue(self, k_all, k_label_all):
        print(f"self.queue_size {self.queue_size}")
        with torch.no_grad():
            k_all = F.normalize(k_all)

            all_size = k_all.shape[0]

            # Process each sample
            for idx in range(all_size):

                current_label = k_label_all[idx].item()
                current_feature = k_all[idx]

                # Initialize dictionaries for new labels if not present
                if current_label not in self.class_indices:
                    self.class_indices[current_label] = 0
                    self.class_positions[current_label] = (
                        []
                    )  # Stores actual memory indices used by this class

                # Get indices for current class in memory bank
                class_mask = self.memory_label == current_label
                indices_of_ones = torch.where(class_mask)[0]  # No need for .int() == 1

                print(f"indices_of_ones {indices_of_ones}")

                memory_idx_to_update = -1  # Initialize with invalid index

                if indices_of_ones.numel() > 0:
                    # Class already exists in memory, find the most similar feature to replace
                    dist = torch.mm(
                        k_all[idx, :].unsqueeze(0),
                        self.memory[indices_of_ones].transpose(1, 0),
                    )
                    # dist shape is [1, num_existing_samples_for_class]

                    # Find the index within indices_of_ones corresponding to the minimum distance (maximum similarity)
                    most_similar_local_idx = torch.argmax(dist, dim=1).item()

                    # Get the actual index in the full memory bank
                    min_idx = indices_of_ones[most_similar_local_idx].item()

                    # print(f"min_idx {min_idx}")
                    memory_idx_to_update = min_idx

                else:
                    # Class is new or all previous entries were overwritten by other classes (if memory is shared dynamically)
                    # Find a slot to insert. Using a simple strategy: fill allocated slots first, then FIFO within class.
                    # This part re-implements the logic from the commented section for adding new elements.

                    if (
                        len(self.class_positions[current_label])
                        < self.max_allowed_samples
                    ):
                        # If class not full based on allocated slots, find the next logical slot
                        # This assumes fixed blocks per class initially, which might not be the case if memory is dynamic.
                        # A safer approach might be needed if memory slots aren't strictly reserved per class.
                        # Let's calculate the intended base index for this class.
                        base_idx = current_label * self.max_allowed_samples
                        pos_in_class_block = len(self.class_positions[current_label])
                        potential_idx = base_idx + pos_in_class_block

                        # Check if this potential index is actually free (might have been taken by another class if memory isn't strictly partitioned)
                        # A simple check: is the label at potential_idx still the initial 0 (or some placeholder)?
                        # This is imperfect. A truly robust dynamic allocation would be more complex.
                        # For now, let's assume we can claim this slot if it's within bounds.
                        if potential_idx < self.queue_size:
                            memory_idx_to_update = potential_idx
                            self.class_positions[current_label].append(
                                memory_idx_to_update
                            )  # Track the used index
                        # else: No free slot in the designated block, fall through to FIFO replacement logic if needed.

                    if (
                        memory_idx_to_update == -1
                        and len(self.class_positions[current_label]) > 0
                    ):
                        # If we couldn't add to a new slot (either block full or issue with finding slot),
                        # and the class has existing entries, use FIFO replacement among its current slots.
                        fifo_replace_local_idx = self.class_indices[current_label]
                        memory_idx_to_update = self.class_positions[current_label][
                            fifo_replace_local_idx
                        ]
                        self.class_indices[current_label] = (
                            self.class_indices[current_label] + 1
                        ) % len(
                            self.class_positions[current_label]
                        )  # Modulo by current count

                    # If memory_idx_to_update is still -1 here, it means the class is new AND its designated block is full
                    # OR the class used to exist but all its slots were overwritten, and now it needs a slot again.
                    # This case needs a strategy: maybe overwrite the oldest entry globally? Or raise error?
                    # Current logic doesn't handle this well. Let's stick to the implemented paths for now.

                # Verify memory_idx is valid and within bounds before updating OR extend if needed
                if (
                    0 <= memory_idx_to_update < self.memory.shape[0]
                ):  # Check against current memory shape
                    self.memory[memory_idx_to_update] = current_feature
                    self.memory_label[memory_idx_to_update] = current_label
                    print(
                        f"Updated memory at index {memory_idx_to_update} with label {current_label}"
                    )
                elif indices_of_ones.numel() == 0 and memory_idx_to_update == -1:
                    # If the item was new and we couldn't find a slot, extend the bank.
                    print(
                        f"No suitable slot found for new label {current_label}. Extending memory bank."
                    )
                    self._extend_single(current_feature, current_label)
                    # The queue size is now larger for subsequent iterations in this batch.
                else:
                    # This case covers:
                    # 1. Existing item where most similar couldn't be replaced (shouldn't happen if min_idx is valid).
                    # 2. New item where extension is not desired/implemented, and no slot was found.
                    print(
                        f"Skipping update for label {current_label}. Could not determine valid index or chose not to extend. "
                        f"indices_of_ones count: {indices_of_ones.numel()}, memory_idx_to_update: {memory_idx_to_update}"
                    )

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
        if self.bank == "memory":
            dist = torch.mm(
                F.normalize(x), self.memory.transpose(1, 0)
            )  # B * Q, memory already normalized
            # print("dist shape", dist.shape)
            max_batch_dist, _ = torch.max(dist, dim=1)
            # print("max_batch_dist ", max_batch_dist)
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
                # print("sim_indices ", sim_indices)
                sim_labels = torch.gather(
                    self.memory_label.expand(x.size(0), -1), dim=-1, index=sim_indices
                )
                # print("sim_labels ", sim_labels)
                # print("sim_labels shape", sim_labels.shape)

                # self.count_labels(sim_labels)

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

        elif self.bank == "anchor":
            print("anchor_memory shape", self.anchor_memory.shape)
            pred_scores = torch.mm(
                F.normalize(x), self.anchor_memory.transpose(1, 0)
            )  # B * C, memory already normalized
            print("pred_scores", pred_scores)
            # print("dist shape", dist.shape)

        return pred_scores

    def build_anchor_bank(self, entropy_list):
        with torch.no_grad():
            for cls in range(self.classes):
                indices = torch.where(self.memory_label == cls)[0]
                # print("indices", indices)
                # print("len(indices)", len(indices))
                # print(
                #     "self.memory[indices].transpose(1, 0) shape",
                #     self.memory[indices].transpose(1, 0).shape,
                # )
                entropy_weight = F.softmax(entropy_list[indices], dim=0)
                # print("mean_feature shape", mean_feature.shape)
                self.anchor_memory[cls] = torch.sum(
                    entropy_weight * self.memory[indices], dim=0
                )

    def ema_update(self, k_all, k_label_all):
        with torch.no_grad():
            unique_labels = torch.unique(k_label_all)
            for label in unique_labels:
                indices = torch.where(k_label_all == label)
                norm_cls_dist_weight = F.softmax(
                    torch.div(
                        1,
                        torch.mm(
                            self.anchor_memory[label], k_all[indices].transpose(1, 0)
                        ),
                    ),
                    dim=-1,
                )  # B * Q, memory already normalized
                self.anchor_memory[label] = (1 - self.alpha) * (
                    torch.sum(norm_cls_dist_weight * k_all[indices], dim=0)
                ) + self.alpha * self.anchor_memory[label]

    def count_labels(self, labels):
        # Adjust labels if they don't start from zero
        labels = labels.view(-1)
        min_label = labels.min()
        adjusted_labels = labels - min_label
        # Count the number of occurrences of each label
        label_counts = torch.bincount(adjusted_labels)

        # display the number of occurrences of each label
        for i, count in enumerate(label_counts):
            print(f"Label {i + min_label}: {count}")


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
