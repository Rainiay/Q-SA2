from __future__ import annotations

import logging
import os
from typing import Callable, Optional, Union

import torch

from peft.import_utils import is_bnb_4bit_available, is_bnb_available


class NFQuantizer:
    def __init__(self, num_bits=2, device="cuda", method="normal", block_size=64, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_bits = num_bits
        self.device = device
        self.method = method
        self.block_size = block_size
        if self.method == "normal":
            self.norm_lookup_table = self.create_normal_map(num_bits=self.num_bits)
            self.norm_lookup_table = self.norm_lookup_table.to(device)
        elif self.method == "uniform":
            self.norm_lookup_table = self.create_uniform_map(num_bits=self.num_bits)
            self.norm_lookup_table = self.norm_lookup_table.to(device)
        else:
            raise NotImplementedError("Other quantization methods not supported yet.")

    @staticmethod
    def create_uniform_map(symmetric=False, num_bits=4):
        if symmetric:
            # print("symmetric uniform quantization")
            negative = torch.linspace(-1, 0, 2 ** (num_bits - 1))
            positive = torch.linspace(0, 1, 2 ** (num_bits - 1))
            table = torch.cat([negative, positive[1:]])
        else:
            # print("asymmetric uniform quantization")
            table = torch.linspace(-1, 1, 2**num_bits)
        return table

    @staticmethod
    def create_normal_map(offset=0.9677083, symmetric=False, num_bits=2):
        try:
            from scipy.stats import norm
        except ImportError:
            raise ImportError("The required package 'scipy' is not installed. Please install it to continue.")

        variations = 2**num_bits
        if symmetric:
            v = norm.ppf(torch.linspace(1 - offset, offset, variations + 1)).tolist()
            values = []
            for index in range(len(v) - 1):
                values.append(0.5 * v[index] + 0.5 * v[index + 1])
            v = values
        else:
            # one more positive value, this is an asymmetric type
            v1 = norm.ppf(torch.linspace(offset, 0.5, variations // 2 + 1)[:-1]).tolist()
            v2 = [0]
            v3 = (-norm.ppf(torch.linspace(offset, 0.5, variations // 2)[:-1])).tolist()
            v = v1 + v2 + v3

        values = torch.Tensor(v)
        values = values.sort().values
        values /= values.max()
        return values

    def quantize_tensor(self, weight):
        max_abs = torch.abs(weight).max()
        weight_normed = weight / max_abs

        weight_normed_expanded = weight_normed.unsqueeze(-1)

        # Reshape L to have the same number of dimensions as X_expanded
        L_reshaped = torch.tensor(self.norm_lookup_table).reshape(1, -1)

        # Calculate the absolute difference between X_expanded and L_reshaped
        abs_diff = torch.abs(weight_normed_expanded - L_reshaped)

        # Find the index of the minimum absolute difference for each element
        qweight = torch.argmin(abs_diff, dim=-1)
        return qweight, max_abs

    def dequantize_tensor(self, qweight, max_abs):
        qweight_flatten = qweight.flatten()

        weight_normed = self.norm_lookup_table[qweight_flatten]
        weight = weight_normed * max_abs

        weight = weight.reshape(qweight.shape)

        return weight

    def quantize_block(self, weight):
        if len(weight.shape) != 2:
            raise ValueError(f"Only support 2D matrix, but your input has {len(weight.shape)} dimensions.")
        if weight.shape[0] * weight.shape[1] % self.block_size != 0:
            raise ValueError(
                f"Weight with shape ({weight.shape[0]} x {weight.shape[1]}) "
                f"is not dividable by block size {self.block_size}."
            )

        M, N = weight.shape
        device = weight.device

        # Quantization
        weight_flatten = weight.flatten()  # (M*N, )
        weight_block = weight_flatten.reshape(-1, self.block_size)  # (L, B), L = M * N / B
        if self.method == "normal":
            weight_max = weight_block.abs().max(dim=-1)[0]  # (L)
        elif self.method == "uniform":
            weight_max = weight_block.mean(dim=-1) + 2.5 * weight_block.std(dim=-1)
        else:
            raise NotImplementedError("Method not supported yet.")
        weight_max = weight_max.unsqueeze(-1)  # (L, 1)
        weight_divabs = weight_block / weight_max  # (L, B)
        weight_divabs = weight_divabs.unsqueeze(-1)  # (L, B, 1)
        L_reshaped = self.norm_lookup_table.reshape(1, -1)  # (1, 2**K)

        abs_diff = torch.abs(weight_divabs - L_reshaped)  # (L, B, 2**K)
        qweight = torch.argmin(abs_diff, dim=-1)  # (L, B)

        # Pack multiple k-bit into uint8
        qweight = qweight.reshape(-1, 8 // self.num_bits)
        qweight_pack = torch.zeros((M * N // 8 * self.num_bits, 1), dtype=torch.uint8, device=device)

        # data format example:
        # [1, 0, 3, 2] or [01, 00, 11, 10]  -> [10110001], LIFO
        for i in range(8 // self.num_bits):
            qweight[:, i] = qweight[:, i] << i * self.num_bits
            qweight_pack[:, 0] |= qweight[:, i]

        return qweight_pack, weight_max, weight.shape

    def dequantize_block(self, qweight, weight_max, weight_shape):
        # unpack weight
        device = qweight.device
        weight = torch.zeros((qweight.shape[0], 8 // self.num_bits), dtype=torch.float32, device=device)
        for i in range(8 // self.num_bits):
            lookup_table_idx = qweight.to(torch.long) % 2**self.num_bits  # get the most right 2 bits
            lookup_table_idx = lookup_table_idx.to(torch.long)
            weight[:, i] = self.norm_lookup_table[lookup_table_idx].squeeze()
            qweight = qweight >> self.num_bits  # right shift 2 bits of the original data

        weight_block = weight.reshape(-1, self.block_size)
        weight = weight_block * weight_max
        weight = weight.reshape(weight_shape)

        return weight


def _low_rank_decomposition(weight, reduced_rank=32):
    """
    :param weight: The matrix to decompose, of shape (H, W) :param reduced_rank: the final rank :return:
    """
    matrix_dimension = len(weight.size())
    if matrix_dimension != 2:
        raise ValueError(f"Only support 2D matrix, but your input has {matrix_dimension} dimensions.")

    # Use SVD to decompose a matrix, default full_matrices is False to save parameters
    U, S, Vh = torch.linalg.svd(weight, full_matrices=False)

    L = U @ (torch.sqrt(torch.diag(S)[:, 0:reduced_rank]))
    R = torch.sqrt(torch.diag(S)[0:reduced_rank, :]) @ Vh

    return {"L": L, "R": R, "U": U, "S": S, "Vh": Vh, "reduced_rank": reduced_rank}


def sparse_lowrank(W, iter=5):
    m, n = W.shape
    lamb = 1.0 / torch.sqrt(torch.tensor(max(m, n), device=W.device))
    S = torch.zeros_like(W)
    Y = W / torch.norm(W)
    mu = 1.25 / torch.norm(W, p=2)
    for _ in range(iter):
        U, S_diag, Vh = torch.linalg.svd(W - S + (1/mu) * Y, full_matrices=False)
        S_shrink = torch.diag(torch.clamp(S_diag - 1/mu, min=0))
        M = U @ S_shrink @ Vh
        S = torch.sign(W - M + (1/mu) * Y) * torch.clamp(torch.abs(W - M + (1/mu) * Y) - lamb/mu, min=0)
        Y = Y + mu * (W - M - S)
    return M, S


@torch.no_grad()
def qsa2_init(weight: Union[torch.Tensor, torch.nn.Parameter], num_bits: int, reduced_rank: int, num_iter=1, de_iter=1):
    if num_iter <= 0:
        raise ValueError("Number of iterations must be greater than 0")

    out_feature, in_feature = weight.size()
    device = weight.device
    dtype = weight.dtype

    logging.info(
        f"Weight: ({out_feature}, {in_feature}) | Rank: {reduced_rank} "
        f"| Num Iter: {num_iter} | Num Bits: {num_bits}"
    )
    quantizer = NFQuantizer(num_bits=num_bits, device=device, method="normal", block_size=32)

    compute_device = device
    weight = weight.to(device=compute_device, dtype=torch.float32)
    M, S = sparse_lowrank(weight, iter=de_iter)
    quantized_weight, max_abs, shape = quantizer.quantize_block(S)
    dequantized_weight = quantizer.dequantize_block(quantized_weight, max_abs, shape)

    # Decompose the residual by SVD
    output = _low_rank_decomposition(weight - dequantized_weight, reduced_rank=reduced_rank)
    L, R, reduced_rank = output["L"], output["R"], output["reduced_rank"]
    for i in range(num_iter-1):
        torch.cuda.empty_cache()
        low_rank_product = L @ R if torch.is_tensor(L) else 0
        residual = weight - low_rank_product
        quantized_weight, max_abs, shape = quantizer.quantize_block(residual)
        dequantized_weight = quantizer.dequantize_block(quantized_weight, max_abs, shape)
        output = _low_rank_decomposition(weight - dequantized_weight, reduced_rank=reduced_rank)
        L, R, reduced_rank = output["L"], output["R"], output["reduced_rank"]
    lora_A, lora_B = R, L

    return dequantized_weight.to(device=device, dtype=dtype), lora_A, lora_B
