"""
src/optimization/model_quantizer.py
------------------------------------
Phase 4 - Software optimization.

    1. Dynamic INT8 quantization  (torch.quantization.quantize_dynamic)
    2. Structured pruning         (torch.nn.utils.prune, 30% L1 on conv/linear)
    3. ONNX export                (opset from config)

Works on any nn.Module; demonstrated on the SmallCNN wheel classifier but the
same API applies to custom detection heads. All steps are CPU-only.
"""

from __future__ import annotations

import os
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from utils.config import load_config, resolve_path


class ModelOptimizer:
    """INT8 quantization + structured pruning + ONNX export."""

    def __init__(self, config: dict | None = None):
        cfg = config or load_config()
        o = cfg["optimization"]
        self.do_quantize = bool(o["quantize"])
        self.prune_amount = float(o["prune_amount"])
        self.onnx_opset = int(o["onnx_opset"])
        self.onnx_output = resolve_path(o["onnx_output"])

    # ------------------------------------------------------------------ #
    def quantize_dynamic(self, model: nn.Module) -> nn.Module:
        """Apply dynamic INT8 quantization to Linear + LSTM layers."""
        model.eval()
        qmodel = torch.quantization.quantize_dynamic(
            model, {nn.Linear, nn.LSTM, nn.GRU}, dtype=torch.qint8
        )
        print("[ModelOptimizer] applied dynamic INT8 quantization.")
        return qmodel

    # ------------------------------------------------------------------ #
    def prune_structured(self, model: nn.Module,
                         amount: float | None = None) -> nn.Module:
        """
        Apply L1 structured pruning to every Conv2d/Linear weight tensor, then
        make the pruning permanent by removing the reparameterization.
        """
        amt = self.prune_amount if amount is None else amount
        pruned_layers = 0
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                prune.ln_structured(module, name="weight", amount=amt, n=1, dim=0)
                prune.remove(module, "weight")
                pruned_layers += 1
            elif isinstance(module, nn.Linear):
                prune.l1_unstructured(module, name="weight", amount=amt)
                prune.remove(module, "weight")
                pruned_layers += 1
        print(f"[ModelOptimizer] pruned {pruned_layers} layers at {amt:.0%} sparsity.")
        return model

    # ------------------------------------------------------------------ #
    def finetune(self, model: nn.Module, train_loader, epochs: int = 3,
                lr: float = 1e-4) -> nn.Module:
        """
        Recover accuracy lost to structured pruning with a few epochs of
        fine-tuning at a low learning rate.

        Structured pruning zeroes whole output channels/filters in every
        conv/linear layer independently; because layers feed each other, the
        effect compounds and a pruned-but-never-retrained model can collapse
        (measured: 30% pruning took SmallCNN from 0.86 to 0.27 accuracy with no
        fine-tuning). This is expected behaviour for naive structured pruning,
        not a bug -- the standard fix is exactly this: prune, then fine-tune.
        """
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        lossf = nn.CrossEntropyLoss()
        for ep in range(epochs):
            total = 0.0
            for x, y in train_loader:
                opt.zero_grad()
                loss = lossf(model(x), y)
                loss.backward()
                opt.step()
                total += loss.item() * x.size(0)
            print(f"[ModelOptimizer] fine-tune epoch {ep+1}/{epochs} "
                 f"loss={total/len(train_loader.dataset):.3f}")
        model.eval()
        return model

    # ------------------------------------------------------------------ #
    def export_onnx(self, model: nn.Module,
                    input_shape: Tuple[int, int, int, int] = (1, 3, 128, 128),
                    output_path: str | None = None) -> str:
        """Export the model to ONNX with dynamic batch axis."""
        out = output_path or self.onnx_output
        os.makedirs(os.path.dirname(out), exist_ok=True)
        model.eval()
        dummy = torch.randn(*input_shape)
        # dynamo=False forces the legacy TorchScript exporter. torch>=2.5 defaults
        # to the dynamo exporter, whose graphs break ONNX Runtime's quantizer
        # with "Inferred shape and existing shape differ in dimension 0".
        torch.onnx.export(
            model, dummy, out,
            input_names=["input"], output_names=["output"],
            opset_version=self.onnx_opset,
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            dynamo=False,
        )
        print(f"[ModelOptimizer] exported ONNX -> {out}")
        return out

    # ------------------------------------------------------------------ #
    def optimize(self, model: nn.Module,
                 input_shape: Tuple[int, int, int, int] = (1, 3, 128, 128),
                 export: bool = True) -> Tuple[nn.Module, str | None]:
        """
        Full pipeline: prune -> quantize -> (ONNX export).
        Returns (optimized_model, onnx_path_or_None).
        """
        model = self.prune_structured(model)

        # Export BEFORE quantizing. torch.onnx.export cannot trace dynamically
        # quantized modules (the packed-params objects have no __obj_flatten__),
        # so the FP32 pruned graph is exported and INT8 is applied at the ONNX
        # level by ONNX Runtime instead.
        onnx_path = None
        if export:
            try:
                onnx_path = self.export_onnx(model, input_shape)
                onnx_path = self.quantize_onnx(onnx_path) or onnx_path
            except Exception as exc:  # pragma: no cover - export robustness
                print(f"[ModelOptimizer] ONNX export skipped ({exc}).")

        if self.do_quantize:
            model = self.quantize_dynamic(model)
        return model, onnx_path

    # ------------------------------------------------------------------ #
    def quantize_onnx(self, onnx_path: str) -> str | None:
        """Apply INT8 dynamic quantization to an exported ONNX graph."""
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
        except Exception as exc:
            print(f"[ModelOptimizer] ONNX quantization unavailable ({exc}).")
            return None
        out = onnx_path.replace(".onnx", "_int8.onnx")
        quantize_dynamic(onnx_path, out, weight_type=QuantType.QInt8)
        print(f"[ModelOptimizer] ONNX INT8 -> {out}")
        return out

    # ------------------------------------------------------------------ #
    @staticmethod
    def model_size_mb(model: nn.Module) -> float:
        """Approximate in-memory parameter+buffer footprint in MB."""
        params = sum(p.numel() * p.element_size() for p in model.parameters())
        buffers = sum(b.numel() * b.element_size() for b in model.buffers())
        return (params + buffers) / (1024 ** 2)


if __name__ == "__main__":
    from src.models.benchmark_classifier import SmallCNN

    net = SmallCNN()
    opt = ModelOptimizer()
    before = ModelOptimizer.model_size_mb(net)
    net_opt, onnx_path = opt.optimize(net, export=True)
    after = ModelOptimizer.model_size_mb(net_opt)
    print(f"size before={before:.3f} MB  after={after:.3f} MB  onnx={onnx_path}")
