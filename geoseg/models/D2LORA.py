import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LoRALayer():
    def __init__(
            self,
            r: int,
            alpha: int,
            dropout: float,
            merge_weights: bool,
    ):
        self.r = r
        self.alpha = alpha
        # Optional dropout
        if dropout > 0.:
            self.dropout = nn.Dropout(p=dropout)
            # self.drop_rate = 1
        else:
            self.dropout = lambda x: x
            # self.drop_rate = 0
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights
        # print(merge_weights)

class D2LoRALinear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
            self,
            in_features: int,
            out_features: int,
            r: int = 0,
            alpha: int = 1,
            dropout: float = 0.,
            fan_in_fan_out: bool = False,
            # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
            merge_weights: bool = True,
            **kwargs
    ):
        nn.Linear.__init__(self, in_features=in_features, out_features=out_features, **kwargs)
        LoRALayer.__init__(self, r=r, alpha=alpha, dropout=dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out

        if self.r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((self.r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, self.r)))
            self.w1 = nn.Parameter(torch.ones(out_features,1))
            self.w2 = nn.Parameter(torch.ones(1, in_features))
            self.weightcopy = 0
            self.scaling = self.alpha/self.r
            self.weight.requires_grad = False

        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)

        if hasattr(self, 'lora_A'):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    BA = self.lora_B @ self.lora_A
                    self.weightcopy = self.w1 * (self.weight + BA) * self.w2#
                    self.weightcopy.to(self.lora_B.device)
                self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            BA = self.lora_B @ self.lora_A
            result = F.linear(x, T(self.w1 * (self.weight + BA) * self.w2), bias=self.bias)
            return result
        if self.r > 0 and self.merged:
            return F.linear(x, T(self.weightcopy), bias=self.bias)
        else:
            return F.linear(x, T(self.weight), bias=self.bias)
