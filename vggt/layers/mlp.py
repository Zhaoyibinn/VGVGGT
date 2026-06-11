# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/mlp.py


from typing import Callable, Optional

import torch
from torch import Tensor, nn

_gelu_fused_warmed = False

@torch.compile
def gelu_fused(x):
    return 0.5 * x * (1 + torch.erf(x * 2**-0.5))

def warmup_gelu_fused():
    global _gelu_fused_warmed
    if _gelu_fused_warmed:
        return
    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        x = torch.rand((1024, 16), device="cuda", dtype=dtype)
        y = gelu_fused(x)
        y = gelu_fused(x)
    _gelu_fused_warmed = True

warmup_gelu_fused()

class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        # self.act = act_layer()
        self.act = gelu_fused
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    @torch.compile
    def forward_infer(self, x):
        x = self.fc1(x)
        x = 0.5 * x * (1 + torch.erf(x * 2**-0.5))
        x = self.fc2(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_infer(x)



class MlpFP32(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        # self.act = act_layer()
        self.act = gelu_fused
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    @staticmethod
    def map_to_args_to_float(args, kwargs):
        args = tuple(
            torch.float32 if isinstance(arg, torch.dtype) else arg
            for arg in args
        )
        kwargs = dict(kwargs)
        for key in kwargs:
            if key == "dtype":
                kwargs[key] = torch.float32
        return args, kwargs

    def to(self, *args, **kwargs):
        self.fc1 = self.fc1.to(*args, **kwargs)
        args, kwargs = self.map_to_args_to_float(args, kwargs)
        self.fc2 = self.fc2.to(*args, **kwargs)
        return self

    @torch.compile
    def forward_infer(self, x):
        x = self.fc1(x)
        x = 0.5 * x * (1 + torch.erf(x * 2**-0.5))
        x = self.fc2(x.float())
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_infer(x)
