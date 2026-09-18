import argparse
import tempfile
from pathlib import Path


def smoke_tenseal() -> None:
    import tenseal as ts

    context = ts.context(ts.SCHEME_TYPE.CKKS, 8192, -1, [60, 40, 40, 60])
    context.global_scale = 2**40
    encrypted = ts.ckks_vector(context, [1.0, 2.0])
    result = (encrypted + encrypted).decrypt()
    assert all(abs(actual - expected) < 1e-3 for actual, expected in zip(result, [2.0, 4.0]))


def smoke_concrete() -> None:
    import numpy as np
    import torch
    from concrete.ml.torch.compile import compile_torch_model

    torch.manual_seed(0)
    model = torch.nn.Linear(4, 2).eval()
    inputset = np.random.default_rng(0).random((20, 4), dtype=np.float32)
    compiled = compile_torch_model(model, inputset, n_bits=4)
    prediction = compiled.forward(inputset[:1], fhe="simulate")
    assert prediction.shape == (1, 2)


def smoke_helayers() -> None:
    import numpy as np
    import pyhelayers
    import tensorflow as tf

    tf.random.set_seed(0)
    model = tf.keras.Sequential(
        [tf.keras.layers.Input(shape=(4,)), tf.keras.layers.Dense(2)]
    )
    model(np.zeros((1, 4), dtype=np.float32))

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        json_path = root / "tiny.json"
        weights_path = root / "tiny.h5"
        json_path.write_text(model.to_json(), encoding="utf-8")
        model.save_weights(weights_path)

        params = pyhelayers.PlainModelHyperParams()
        plain = pyhelayers.NeuralNetPlain()
        plain.init_from_files(params, [str(json_path), str(weights_path)])
        requirements = pyhelayers.HeRunRequirements()
        requirements.set_he_context_options([pyhelayers.DefaultContext()])
        requirements.optimize_for_batch_size(1)
        profile = pyhelayers.HeModel.compile(plain, requirements)
        context = pyhelayers.HeModel.create_context(profile)
        assert context is not None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("backend", choices=("tenseal", "concrete", "helayers"))
    args = parser.parse_args()
    globals()[f"smoke_{args.backend}"]()
    print(f"{args.backend}: smoke test passed")


if __name__ == "__main__":
    main()
