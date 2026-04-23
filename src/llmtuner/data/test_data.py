"""Lightweight smoke-test entrypoint for dataset loading."""


def _not_configured() -> None:
    raise RuntimeError(
        "This helper requires runtime objects: dataset_attr, model_args, and data_args. "
        "Import `load_single_dataset` and call it from your training/eval script."
    )


if __name__ == "__main__":
    _not_configured()
