"""Compile each real HEDiff architecture and execute one encrypted input."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch


def load_upstream(root: Path):
    sys.path.insert(0, str(root))
    from plain_models import CryptoNet_Digits, CryptoNet_MNIST, MLP_Bank, MLP_Credit

    return CryptoNet_Digits, CryptoNet_MNIST, MLP_Bank, MLP_Credit


def torch_model(root: Path, dataset: str):
    digits_cls, mnist_cls, bank_cls, credit_cls = load_upstream(root)
    specs = {
        "credit": (credit_cls, (64, 23)),
        "bank": (bank_cls, (64, 20)),
        "digits": (digits_cls, (64, 1, 8, 8)),
        "mnist": (mnist_cls, (64, 1, 28, 28)),
    }
    model_cls, shape = specs[dataset]
    torch.manual_seed(42)
    return model_cls().eval(), np.random.default_rng(42).random(shape, dtype=np.float32)


def summarize(setup_seconds: float, samples: list[float]) -> dict[str, object]:
    return {
        "setup_seconds": setup_seconds,
        "inference_seconds": samples,
        "inference_mean_seconds": statistics.mean(samples),
        "inference_median_seconds": statistics.median(samples),
    }


def run_tenseal(root: Path, dataset: str, repeats: int) -> dict[str, object]:
    import tenseal as ts

    sys.path.insert(0, str(root))
    from base_ts import (
        BankMLP_TS,
        CreditMLP_TS,
        DigitsCryptoNet_TS,
        MNISTCryptoNet_TS,
        PredictConvEncVector,
        PredictEncVector,
    )

    model, calibration = torch_model(root, dataset)
    setup_started = time.monotonic()
    bits = 26
    degree = 2**14 if dataset in {"digits", "mnist"} else 2**13
    middle = 8 if dataset in {"digits", "mnist"} else 6
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=degree,
        coeff_mod_bit_sizes=[bits + 5] + [bits] * middle + [bits + 5],
    )
    context.global_scale = 2**bits
    context.generate_galois_keys()
    wrappers = {
        "credit": CreditMLP_TS,
        "bank": BankMLP_TS,
        "digits": DigitsCryptoNet_TS,
        "mnist": MNISTCryptoNet_TS,
    }
    encrypted_model = wrappers[dataset](model)
    setup_seconds = time.monotonic() - setup_started
    sample = torch.from_numpy(calibration[:1])
    durations = []
    for _ in range(repeats):
        started = time.monotonic()
        if dataset in {"digits", "mnist"}:
            output, label = PredictConvEncVector(
                encrypted_model, sample, context, model.conv1.kernel_size, model.conv1.stride[0]
            )
        else:
            output, label = PredictEncVector(encrypted_model, sample, context)
        durations.append(time.monotonic() - started)
    print(f"decrypted_shape={tuple(output.shape)}, label={label.item()}")
    return summarize(setup_seconds, durations)


def run_concrete(root: Path, dataset: str, repeats: int) -> dict[str, object]:
    from concrete.ml.torch.compile import compile_torch_model

    model, calibration = torch_model(root, dataset)
    setup_started = time.monotonic()
    compiled = compile_torch_model(
        model,
        calibration,
        n_bits=6,
        rounding_threshold_bits=8,
        p_error=0.01,
    )
    setup_seconds = time.monotonic() - setup_started
    # Exclude one-time key generation and runtime initialization from steady-state inference.
    compiled.forward(calibration[:1], fhe="execute")
    durations = []
    for _ in range(repeats):
        started = time.monotonic()
        output = compiled.forward(calibration[:1], fhe="execute")
        durations.append(time.monotonic() - started)
    print(f"decrypted_shape={output.shape}, label={int(np.argmax(output))}")
    return summarize(setup_seconds, durations)


def run_helayers(root: Path, dataset: str, repeats: int) -> dict[str, object]:
    import pyhelayers
    import tensorflow as tf

    rng = np.random.default_rng(42)
    with tempfile.TemporaryDirectory() as directory:
        directory_path = Path(directory)
        files: list[str]
        if dataset in {"digits", "mnist"}:
            sys.path.insert(0, str(root))
            from plain_models_tf import CryptoNet_DIGITS_tf_poly, CryptoNet_MNIST_tf

            model = CryptoNet_DIGITS_tf_poly() if dataset == "digits" else CryptoNet_MNIST_tf()
            shape = (1, 8, 8, 1) if dataset == "digits" else (1, 28, 28, 1)
            sample = rng.random(shape, dtype=np.float32)
            model(sample)
            json_path = directory_path / f"{dataset}.json"
            weights_path = directory_path / f"{dataset}.h5"
            json_path.write_text(model.to_json(), encoding="utf-8")
            model.save_weights(weights_path)
            files = [str(json_path), str(weights_path)]
        else:
            sys.path.insert(0, str(root))
            from fhe_onnx_convert import MLP_Credit
            from plain_models import MLP_Bank

            model = (MLP_Credit() if dataset == "credit" else MLP_Bank()).eval()
            width = 23 if dataset == "credit" else 20
            sample = rng.random((1, width), dtype=np.float32)
            onnx_path = directory_path / f"{dataset}.onnx"
            torch.onnx.export(
                model,
                torch.from_numpy(sample),
                onnx_path,
                input_names=["inputs"],
                output_names=["outputs"],
            )
            files = [str(onnx_path)]

        setup_started = time.monotonic()
        params = pyhelayers.PlainModelHyperParams()
        plain = pyhelayers.NeuralNetPlain()
        plain.init_from_files(params, files)
        requirements = pyhelayers.HeRunRequirements()
        requirements.set_he_context_options([pyhelayers.DefaultContext()])
        requirements.optimize_for_batch_size(1)
        profile = pyhelayers.HeModel.compile(plain, requirements)
        context = pyhelayers.HeModel.create_context(profile)
        encrypted_model = pyhelayers.NeuralNet(context)
        encrypted_model.encode_encrypt(plain, profile)
        processor = encrypted_model.create_model_io_encoder()
        setup_seconds = time.monotonic() - setup_started
        durations = []
        for _ in range(repeats):
            started = time.monotonic()
            encrypted_input = pyhelayers.EncryptedData(context)
            processor.encode_encrypt(encrypted_input, [sample])
            encrypted_output = pyhelayers.EncryptedData(context)
            encrypted_model.predict(encrypted_output, encrypted_input)
            output = processor.decrypt_decode_output(encrypted_output)
            durations.append(time.monotonic() - started)
        print(f"decrypted_shape={np.asarray(output).shape}, label={int(np.argmax(output))}")
        return summarize(setup_seconds, durations)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("backend", choices=("tenseal", "concrete", "helayers"))
    parser.add_argument("dataset", choices=("credit", "bank", "digits", "mnist"))
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    started = time.monotonic()
    result = globals()[f"run_{args.backend}"](
        args.source_root.resolve(), args.dataset, args.repeats
    )
    result.update({"backend": args.backend, "dataset": args.dataset})
    print("BENCHMARK_JSON=" + json.dumps(result, sort_keys=True))
    print(
        f"PASS backend={args.backend} dataset={args.dataset} "
        f"elapsed_seconds={time.monotonic() - started:.2f}"
    )


if __name__ == "__main__":
    main()
