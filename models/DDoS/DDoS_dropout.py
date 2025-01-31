from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class Dropout(nn.Module):
    """Dropout module.

    Implementation of dropout with the ability to share the dropout mask
    along a particular dimension.

    If not in training mode, this module computes the identity function.

    Supplementary '1.11.6 Dropout details'.

    Args:
        p: Dropout rate (probability of an element to be zeroed).
        share_dim: Dimension(s) along which the dropout mask is shared.
        inplace: If set to `True`, will do this operation in-place.

    """

    def __init__(
        self,
        p: float,
        share_dim: Union[int, Tuple[int, ...]] = (),
        inplace: bool = False,
    ) -> None:
        super(Dropout, self).__init__()
        assert 0.0 <= p <= 1.0
        self.p = p
        if type(share_dim) == int:
            share_dim = (share_dim,)
        else:
            assert isinstance(share_dim, tuple)
        self.share_dim = share_dim
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = list(x.shape)
        for d in self.share_dim:
            shape[d] = 1
        mask = x.new_ones(shape)
        mask = F.dropout(
            input=mask,
            p=self.p,
            training=self.training,
            inplace=self.inplace,
        )
        x *= mask
        return x


class DropoutRowwise(nn.Module):
    """Applies dropout to entire rows of the input tensor.
    
    Args:
        p: Dropout probability (0 to 1).
    """
    def __init__(self, p: float = 0.5):
        super(DropoutRowwise, self).__init__()
        self.p = p
        
    def forward(self, x):
        if not self.training or self.p == 0:
            return x
            
        shape = list(x.shape)
        device = x.device
        
        # Create mask that drops entire rows
        mask_shape = [shape[0], 1]
        if len(shape) > 2:
            mask_shape.extend([1] * (len(shape) - 2))
            
        mask = torch.bernoulli(torch.full(mask_shape, 1 - self.p, device=device))
        mask = mask.expand_as(x)
        
        return x * mask / (1 - self.p)


class DropoutColumnwise(nn.Module):
    """Applies dropout to entire columns of the input tensor.
    
    Args:
        p: Dropout probability (0 to 1).
    """
    def __init__(self, p: float = 0.5):
        super(DropoutColumnwise, self).__init__()
        self.p = p
        
    def forward(self, x):
        if not self.training or self.p == 0:
            return x
            
        shape = list(x.shape)
        device = x.device
        
        # Create mask that drops entire columns
        mask_shape = [1, shape[1]]
        if len(shape) > 2:
            mask_shape.extend([1] * (len(shape) - 2))
            
        mask = torch.bernoulli(torch.full(mask_shape, 1 - self.p, device=device))
        mask = mask.expand_as(x)
        
        return x * mask / (1 - self.p)