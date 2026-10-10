# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Regression tests for native CPU grouped-query SDPA and its fallback."""

import unittest
from unittest.mock import patch

from parameterized import parameterized

from transformers import is_torch_available
from transformers.testing_utils import require_torch
from transformers.utils.import_utils import is_torch_greater_or_equal


if is_torch_available():
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    from transformers.integrations import sdpa_attention


@require_torch
class CpuGqaSdpaTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.module = torch.nn.Module()
        self.module.num_key_value_groups = 3
        self.module.is_causal = True

    def _inputs(self, dtype=None, q_length=3, head_dim=64, value_dim=None):
        if dtype is None:
            dtype = torch.float32
        query = torch.randn(2, 9, q_length, head_dim, dtype=dtype)
        key = torch.randn(2, 3, 17, head_dim, dtype=dtype)
        value = torch.randn(2, 3, 17, value_dim or head_dim, dtype=dtype)
        return query, key, value

    def _mask(self, kind, q_length, dtype):
        if kind == "none":
            return None
        mask = torch.rand(2, 1, q_length, 17) > 0.25
        if kind == "boolean":
            return mask
        return torch.zeros_like(mask, dtype=dtype).masked_fill(~mask, float("-inf"))

    def _reference(self, query, key, value, mask, **kwargs):
        return (
            torch.nn.functional.scaled_dot_product_attention(
                query,
                key.repeat_interleave(3, dim=1),
                value.repeat_interleave(3, dim=1),
                attn_mask=mask,
                is_causal=query.shape[2] > 1 and mask is None,
                **kwargs,
            )
            .transpose(1, 2)
            .contiguous()
        )

    @parameterized.expand(
        [
            (dtype, mask, q_length)
            for dtype in ("float32", "float64", "float16", "bfloat16")
            for mask in ("none", "boolean", "additive")
            for q_length in (1, 3)
        ]
    )
    @unittest.skipUnless(is_torch_greater_or_equal("2.9", accept_dev=False), "CPU flash GQA requires torch >= 2.9")
    def test_native_cpu_flash_matches_repeated_kv(self, dtype, mask, q_length):
        query, key, value = self._inputs(getattr(torch, dtype), q_length)
        attention_mask = self._mask(mask, q_length, query.dtype)
        sdpa = torch.nn.functional.scaled_dot_product_attention
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            expected = self._reference(query, key, value, attention_mask)
            with patch("torch.nn.functional.scaled_dot_product_attention", wraps=sdpa) as call:
                actual, weights = sdpa_attention.sdpa_attention_forward(self.module, query, key, value, attention_mask)
        # CPU flash may use different blocking for compact and repeated KV.
        # Use the dtype tolerances from the model SDPA tests and check both paths
        # against an independent higher-precision math result.
        atol, rtol = {
            "float32": (1e-6, 1e-5),
            "float64": (1e-7, 1e-7),
            "float16": (1e-3, 1e-3),
            "bfloat16": (1e-2, 1e-2),
        }[dtype]
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        oracle_mask = attention_mask
        if oracle_mask is not None and oracle_mask.dtype != torch.bool:
            oracle_mask = oracle_mask.double()
        with sdpa_kernel(SDPBackend.MATH):
            oracle = self._reference(query.double(), key.double(), value.double(), oracle_mask)
        torch.testing.assert_close(actual.double(), oracle, atol=atol, rtol=rtol)
        torch.testing.assert_close(expected.double(), oracle, atol=atol, rtol=rtol)
        self.assertIsNone(weights)
        self.assertEqual(call.call_count, 1)
        self.assertEqual(call.call_args.args[1].shape[1], 3)
        self.assertTrue(call.call_args.kwargs["enable_gqa"])

    @parameterized.expand([("float32",), ("float64",), ("float16",), ("bfloat16",)])
    @unittest.skipUnless(is_torch_greater_or_equal("2.9", accept_dev=False), "CPU flash GQA requires torch >= 2.9")
    def test_fully_masked_rows_are_zero(self, dtype):
        query, key, value = self._inputs(getattr(torch, dtype))
        mask = torch.zeros(2, 1, 3, 17, dtype=torch.bool)
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            actual, _ = sdpa_attention.sdpa_attention_forward(self.module, query, key, value, mask)
        torch.testing.assert_close(actual, torch.zeros_like(actual), rtol=0, atol=0)

    @parameterized.expand([("old_runtime", False, 64), ("unequal_value_dim", True, 32)])
    def test_repeated_kv_fallback(self, name, enabled, value_dim):
        query, key, value = self._inputs(value_dim=value_dim)
        mask = self._mask("boolean", 3, query.dtype)
        expected = self._reference(query, key, value, mask)
        sdpa = torch.nn.functional.scaled_dot_product_attention
        with (
            patch.object(sdpa_attention, "_is_torch_xpu_available", False),
            patch.object(sdpa_attention, "_is_torch_mps_available", False),
            patch.object(sdpa_attention, "_is_torch_greater_or_equal_than_2_9", enabled),
        ):
            with patch("torch.nn.functional.scaled_dot_product_attention", wraps=sdpa) as call:
                actual, _ = sdpa_attention.sdpa_attention_forward(self.module, query, key, value, mask)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(call.call_args.args[1].shape[1], 9)
        self.assertFalse(call.call_args.kwargs.get("enable_gqa", False))

    @parameterized.expand([("none",), ("boolean",), ("additive",)])
    def test_math_backward_matches_repeated_kv(self, mask):
        query, key, value = self._inputs()
        attention_mask = self._mask(mask, 3, query.dtype)
        inputs = [tensor.requires_grad_() for tensor in (query, key, value)]
        with sdpa_kernel(SDPBackend.MATH):
            expected = self._reference(*inputs, attention_mask)
            actual, _ = sdpa_attention.sdpa_attention_forward(self.module, *inputs, attention_mask)
            expected_grads = torch.autograd.grad(expected.square().sum(), inputs)
            actual_grads = torch.autograd.grad(actual.square().sum(), inputs)
        torch.testing.assert_close(actual, expected)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad)

    @parameterized.expand(
        [
            (dtype, mask)
            for dtype in ("float32", "float64", "float16", "bfloat16")
            for mask in ("none", "boolean", "additive")
        ]
    )
    @unittest.skipUnless(is_torch_greater_or_equal("2.9", accept_dev=False), "CPU flash GQA requires torch >= 2.9")
    def test_native_cpu_flash_backward_matches_repeated_kv(self, dtype, mask):
        query, key, value = self._inputs(getattr(torch, dtype))
        attention_mask = self._mask(mask, 3, query.dtype)
        inputs = [tensor.requires_grad_() for tensor in (query, key, value)]
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            expected = self._reference(*inputs, attention_mask)
            actual, _ = sdpa_attention.sdpa_attention_forward(self.module, *inputs, attention_mask)
            probe = torch.randn_like(expected)
            expected_grads = torch.autograd.grad((expected * probe).sum(), inputs)
            actual_grads = torch.autograd.grad((actual * probe).sum(), inputs)
        atol, rtol = {
            "float32": (2e-6, 2e-5),
            "float64": (1e-10, 1e-10),
            "float16": (2e-3, 1e-2),
            "bfloat16": (2e-2, 5e-2),
        }[dtype]
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad, atol=atol, rtol=rtol)

    @unittest.skipUnless(is_torch_greater_or_equal("2.9", accept_dev=False), "CPU flash GQA requires torch >= 2.9")
    def test_large_cpu_head_dimension_uses_native_flash(self):
        query, key, value = self._inputs(q_length=1, head_dim=320)
        mask = self._mask("boolean", 1, query.dtype)
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            expected = self._reference(query, key, value, mask)
            actual, _ = sdpa_attention.sdpa_attention_forward(self.module, query, key, value, mask)
        torch.testing.assert_close(actual, expected)
        self.assertTrue(sdpa_attention.use_gqa_in_sdpa(mask, key, value))

    @unittest.skipUnless(is_torch_greater_or_equal("2.9", accept_dev=False), "CPU flash GQA requires torch >= 2.9")
    def test_cpu_dispatch_with_accelerator_available(self):
        _, key, value = self._inputs()
        mask = self._mask("boolean", 3, key.dtype)
        with patch.object(sdpa_attention, "_is_torch_xpu_available", True):
            with patch.object(sdpa_attention, "_is_torch_greater_or_equal_than_2_8", False):
                self.assertTrue(sdpa_attention.use_gqa_in_sdpa(mask, key, value))

    def _blocked_backward_inputs(self, dtype):
        torch.manual_seed(0)
        query = torch.randn(1, 16, 257, 64, dtype=dtype)
        key = torch.randn(1, 1, 65, 64, dtype=dtype)
        value = torch.randn(1, 1, 65, 64, dtype=dtype)
        mask = torch.rand(1, 1, 257, 65) > 0.2
        probe = torch.randn(1, 257, 16, 64, dtype=dtype)
        return query, key, value, mask, probe

    def _assert_blocked_backward(self, fn, dtype, capture=False):
        query, key, value, mask, probe = self._blocked_backward_inputs(dtype)
        self.module.num_key_value_groups = 16
        self.module.is_causal = False
        inputs = [x.requires_grad_() for x in (query, key, value)]
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            expected = (
                torch.nn.functional.scaled_dot_product_attention(
                    query, key.repeat_interleave(16, 1), value.repeat_interleave(16, 1), attn_mask=mask
                )
                .transpose(1, 2)
                .contiguous()
            )
            actual = fn(query, key, value, mask)
            expected_grads = torch.autograd.grad((expected * probe).sum(), inputs)
            actual_grads = torch.autograd.grad((actual * probe).sum(), inputs)
        # Eager fallback must preserve stock exactly; captured programs use
        # the unchanged dtype tolerances of the existing backward regressions.
        atol, rtol = ((0.02, 0.05) if dtype == torch.bfloat16 else (0.002, 0.01)) if capture else (0, 0)
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad, atol=atol, rtol=rtol)

    @parameterized.expand([("float16",), ("bfloat16",)])
    @unittest.skipUnless(is_torch_greater_or_equal("2.9", accept_dev=False), "CPU flash requires torch >= 2.9")
    def test_blocked_backward_preserves_stock(self, dtype):
        self._assert_blocked_backward(
            lambda q, k, v, mask: sdpa_attention.sdpa_attention_forward(self.module, q, k, v, mask)[0],
            getattr(torch, dtype),
        )

    def _capture_module(self, groups=16):
        class Attention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.num_key_value_groups = groups
                self.is_causal = False

            def forward(self, q, k, v, mask):
                return sdpa_attention.sdpa_attention_forward(self, q, k, v, mask)[0]

        return Attention()

    @parameterized.expand([("export",), ("jit",)])
    @unittest.skipUnless(is_torch_greater_or_equal("2.9", accept_dev=False), "CPU flash requires torch >= 2.9")
    def test_capture_then_backward_preserves_stock(self, capture):
        sample = self._blocked_backward_inputs(torch.bfloat16)[:4]
        module = self._capture_module()
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            if capture == "export":
                captured = torch.export.export(module, sample, strict=True).module()
            else:
                captured = torch.jit.trace(module, sample)
        self._assert_blocked_backward(captured, torch.bfloat16, capture=True)

    @unittest.skipUnless(is_torch_greater_or_equal("2.9", accept_dev=False), "Export requires torch >= 2.9")
    def test_export_independent_value_width(self):
        import io

        module = self._capture_module(groups=3)
        sample = (
            torch.randn(2, 9, 3, 16),
            torch.randn(2, 3, 11, 16),
            torch.randn(2, 3, 11, 16),
            torch.rand(2, 1, 3, 11) > 0.2,
        )
        width = torch.export.Dim("value_width", min=8, max=32)
        with sdpa_kernel(SDPBackend.MATH):
            program = torch.export.export(module, sample, dynamic_shapes=({}, {}, {3: width}, {}), strict=True)
            buffer = io.BytesIO()
            torch.export.save(program, buffer)
            buffer.seek(0)
            replay = torch.export.load(buffer).module()
            for value_width in (16, 24):
                q, k, _, mask = sample
                v = torch.randn(2, 3, 11, value_width)
                expected = (
                    torch.nn.functional.scaled_dot_product_attention(
                        q, k.repeat_interleave(3, 1), v.repeat_interleave(3, 1), attn_mask=mask
                    )
                    .transpose(1, 2)
                    .contiguous()
                )
                torch.testing.assert_close(replay(q, k, v, mask), expected)
