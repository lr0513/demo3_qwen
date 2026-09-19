import argparse

from src.config import ExperimentConfig
from src.evaluator import evaluate_adapter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = ExperimentConfig.from_file(args.config)
    model_path = args.model_path or config.require("model_path")
    data_dir = config.resolve_path(config.require("data_dir"))
    output_dir = config.resolve_path(config.require("output_dir"))
    adapter = args.adapter or str(output_dir / "best")
    output = args.output or str(output_dir / f"eval_{args.split}.json")
    limit = args.limit
    if limit is None:
        limit = config.get("test_limit")

    result = evaluate_adapter(
        base_model_path=model_path,
        adapter_path=adapter,
        data_path=data_dir,
        data_dir=data_dir,
        instruction=config.require("instruction"),
        method=config.require("method"),
        config=config,
        split=args.split,
        output_path=output,
        limit=limit,
    )
    print(
        f"{args.split}: P={result['precision']:.4f}, "
        f"R={result['recall']:.4f}, F1={result['f1']:.4f}"
    )


if __name__ == "__main__":
    main()

