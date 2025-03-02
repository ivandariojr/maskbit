"""This file contains the definition of the look-free quantizer."""

import math
from typing import Mapping, Text, Tuple

import torch
from torch import nn
from einops import rearrange, reduce
from typing import Literal
from maskbit.modeling.quantizer.quantizer_utils import entropy_loss_fn, clamp_log


class LookupFreeQuantizer(torch.nn.Module):
    def __init__(
        self,
        strategy: Literal['default', 'bern', 'bsq'] = 'default',
        ent_strategy: Literal['none', 'default', 'bern', 'bsq'] = 'default',
        token_bits: int = 10,
        commitment_cost: float = 0.25,
        entropy_loss_weight: float = 0.1,
        entropy_loss_temperature: float = 0.01,
        entropy_gamma: float = 1.0,
        allocate_codes: bool = True,
    ):
        """ Initializes the lookup-free quantizer.

        Args:
            token_bits -> int: The number of bits per token.
            commitment_cost -> float: The commitment cost.
            entropy_loss_weight -> float: The weight of the entropy loss.
            entropy_loss_temperature -> float: The temperature for the entropy loss.
            entropy_gamma -> float: The gamma for the entropy loss.
            allocate_codes -> bool: Whether to allocate codes or not.
        """
        super().__init__()
        self.token_size = token_bits
        self.codebook_size = 2 ** token_bits
        self.ent_strategy = ent_strategy
        self.commitment_cost = commitment_cost
        self.entropy_loss_weight = entropy_loss_weight
        self.entropy_loss_temperature = entropy_loss_temperature
        self.entropy_gamma = entropy_gamma

        bits_to_indices = 2 ** torch.arange(0, token_bits, dtype=torch.long)
        self.register_buffer('bits_to_indices', bits_to_indices.long())

        self.allocate_codes = allocate_codes
        if self.allocate_codes:
            all_codes = torch.arange(self.codebook_size)
            bits = ((all_codes[..., None].long() & self.bits_to_indices) != 0).float()
            self.register_buffer('codebook', bits * 2.0 - 1.0)
        self.strategy = strategy
        if self.strategy == "bern":
            self.bn = nn.BatchNorm1d(self.token_size)


    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        """ Forward pass of the quantizer.

        Args:
            z -> torch.Tensor: The input tensor.

        Returns:
            z_quantized -> torch.Tensor: The quantized latent representation.
            result_dict -> Mapping[Text, torch.Tensor]: A dictionary containing additional results
                and losses from the quantizer.
        """
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        if self.strategy == "default":
            ones = torch.ones_like(z)
            sign_mask = (z > 0.0)
            z_quantized = torch.where(sign_mask, ones, -ones)
        elif self.strategy == "bern":
            z_shape = z.shape
            z_flat = z.reshape(-1, z.shape[-1])
            z_bn = self.bn(z_flat).reshape(z_shape)
            # p = torch.sigmoid(z_bn)
            # z = z_bn
            p = (torch.sin(z_bn) + 1) / 2.0 # we need the probabilities  later
            z = 2*p - 1 # TODO: actually sin(z_bn)
            if self.training:
                u = torch.rand_like(p)
                samples = (u < p).float()
                z_quantized = 2 * samples -1
            else:
                z_quantized = torch.sign(z)
        elif self.strategy == "bsq":
            z = z / torch.norm(z, dim=-1, keepdim=True)
            z_quantized = torch.where(
                z > 0,
                torch.tensor(1, dtype=z.dtype, device=z.device),
                torch.tensor(-1, dtype=z.dtype, device=z.device),
            )

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        min_encoding_indices = self.convert_bits_to_indices(z_quantized)

        # compute loss for embedding
        entropy_loss = torch.zeros((), device=z.device)
        per_sample_entropy = torch.zeros((), device=z.device)
        avg_entropy = torch.zeros((), device=z.device)

        # Use entropy loss on the codebook
        match self.ent_strategy:
            case "default":
                if self.entropy_loss_weight != 0.0 and self.training and self.allocate_codes:
                    d = - 2 * torch.einsum('b h w c, n c -> b h w n', z, self.codebook)

                    per_sample_entropy, avg_entropy = entropy_loss_fn(-1*d, self.entropy_loss_temperature, self.entropy_gamma)
                    entropy_loss = self.entropy_loss_weight * (per_sample_entropy - avg_entropy)
                    
            case "bern":
                # step 1: compute per sample entropy
                assert self.strategy == "bern", "Bernoulli entropy loss only works with bernoulli quantizer"
                probs = torch.stack([p, 1 - p], dim=-1)  # (B, H, W, num_bits, 2)
                per_sample_entropy = -torch.mean(
                    torch.sum(probs * clamp_log(probs), dim=-1)  # (B, H, W, num_bits)
                )  # term 1, (,)

                # step 2: compute average entropy
                avg_probs = torch.mean(probs, dim=list(range(probs.ndim - 2)))  # (num_bits, 2)
                avg_entropy = -torch.sum(avg_probs * clamp_log(avg_probs))  # term 2, (,)

                entropy_loss = self.entropy_loss_weight * (per_sample_entropy - self.entropy_gamma * avg_entropy)

            case "bsq":
                # from: https://github.com/zhaoyue-zephyrus/bsq-vit/blob/4ba8afa6829b6a8fa17f6d5d5fdf860436f9611b/transcoder/models/quantizer/bsq.py#L121-L145
                # * assume group size is 1
                # * assume inv_temperature is 1
                # * assume mode is "analytical"
                
                # step 1: compute per sample entropy
                p = torch.sigmoid(-4 * z / math.sqrt(self.token_size))
                probs = torch.stack([p, 1 - p], dim=-1)
                per_sample_entropy = -torch.mean(
                    torch.sum(probs * clamp_log(probs), dim=-1)
                )

                # step 2: compute average entropy
                avg_probs = torch.mean(probs, dim=list(range(probs.ndim - 2)))  # (num_bits, 2)
                avg_entropy = -torch.sum(avg_probs * clamp_log(avg_probs))  # (,)

                entropy_loss = self.entropy_loss_weight * (per_sample_entropy - self.entropy_gamma * avg_entropy)

            case "none":
                pass

        # preserve gradients
        z_quantized = z + (z_quantized - z).detach()

        if self.strategy == "bsq":
            z_quantized = z_quantized / math.sqrt(self.token_size)

        commitment_loss = self.commitment_cost * torch.mean((z_quantized.detach() - z) **2)
        loss = commitment_loss + entropy_loss

        # reshape back to match original input shape
        z_quantized = rearrange(z_quantized, 'b h w c -> b c h w').contiguous()

        result_dict = dict(
            quantizer_loss=loss,
            commitment_loss=commitment_loss,
            entropy_loss=entropy_loss,
            per_sample_entropy=per_sample_entropy,
            avg_entropy=avg_entropy,
            min_encoding_indices=min_encoding_indices
        )

        return z_quantized, result_dict

    def get_codebook_entry(self, indices: torch.Tensor) -> torch.Tensor:
        """ Returns the `codebook entry` for the given indices.

        As the codebook exists only implicitly, this is mainly an integer conversion to a bit representation.
        Note: The bits are represented by {-1, 1}.

        Args:
            indices -> torch.Tensor: The indices in range 0 to codebook size - 1.

        Returns:
            tokens -> torch.Tensor: The bit representation.
        """
        indices = indices.long()
        bits = ((indices[..., None].long() & self.bits_to_indices) != 0).float()
        tokens = bits * 2.0 - 1.0  # scale to -1..1
        return tokens

    def convert_bits_to_indices(self, tokens: torch.Tensor) -> torch.Tensor:
        """ Converts the given tokens to index numbers.

        As the codebook exists only implicitly, this is mainly an integer conversion from a bit representation.
        Note: The bits are represented by {-1, 1}.

        Args:
            tokens -> torch.Tensor: The tokens.

        Returns:
            indices -> torch.Tensor: The indices in range 0 to codebook size - 1.
        """
        tokens = rearrange(tokens, 'b h w c -> b h w c').contiguous()
        sign_mask = (tokens > 0.0)
        return reduce(sign_mask.long() * self.bits_to_indices, 'b h w c -> b h w', 'sum')

    def convert_indices_to_bits(self, indices: torch.Tensor) -> torch.Tensor:
        """ Converts the given indices to tokens.

        As the codebook exists only implicitly, this is mainly an integer conversion to a bit representation.
        Note: The bits are represented by {-1, 1}.

        Args:
            indices -> torch.Tensor: The indices in range 0 to codebook size - 1.

        Returns:
            tokens -> torch.Tensor: The bit representation.
        """
        indices = indices.long()
        return self.get_codebook_entry(indices)



if  __name__ == "__main__":
    quantizer = LookupFreeQuantizer(
        token_bits=10,
        commitment_cost=0.25,
        entropy_loss_weight=0.1,
        entropy_loss_temperature=0.01,
        entropy_gamma=1.0
    )
    all_entries = torch.arange(1024).reshape(1, 1, 1024)
    indices = quantizer.convert_bits_to_indices(quantizer.convert_indices_to_bits(all_entries))
    assert torch.equal(
        indices,
        all_entries
    )
    assert torch.equal(
        quantizer.convert_bits_to_indices(quantizer.codebook.reshape(1,1,1024,10)),
        all_entries
    )