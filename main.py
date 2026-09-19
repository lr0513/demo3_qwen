import argparse
from pathlib import Path

from src.config import ExperimentConfig
from src.data import BC2GMDataset, CausalLMCollator
from src.model import build_training_model, load_tokenizer
from src.trainer import NERTrainer
from src.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--train_limit", type=int, default=None)
    parser.add_argument("--dev_limit", type=int, default=None)
    parser.add_argument("--test_limit", type=int, default=None)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--disable_swanlab", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = ExperimentConfig.from_file(args.config)

    if args.model_path:
        config.set("model_path", args.model_path)
    if args.output_dir:
        config.set("output_dir", args.output_dir)
    if args.test_limit is not None:
        config.set("test_limit", args.test_limit)
    if args.max_epochs is not None:
        config.set("num_epochs", args.max_epochs)
    if args.disable_swanlab:
        config.set("use_swanlab", False)

    set_seed(config.get("seed", 42))
    data_dir = config.resolve_path(config.require("data_dir"))
    output_dir = config.resolve_path(config.require("output_dir"))
    config.set("output_dir", str(output_dir))

    instruction = config.require("instruction")
    tokenizer = load_tokenizer(config.require("model_path"))
    collator = CausalLMCollator(
        tokenizer=tokenizer,
        instruction=instruction,
        max_length=config.get("max_length", 256),
        use_sft_mask=True,
    )

    train_dataset = BC2GMDataset(
        Path(data_dir) / "train.json",
        instruction,
        max_samples=args.train_limit,
    )
    dev_dataset = BC2GMDataset(
        Path(data_dir) / "dev.json",
        instruction,
        max_samples=args.dev_limit,
    )
    test_dataset = BC2GMDataset(
        Path(data_dir) / "test.json",
        instruction,
    )
    print(
        f"Dataset sizes: train={len(train_dataset)}, "
        f"dev={len(dev_dataset)}, test={len(test_dataset)}"
    )

    model = build_training_model(
        config.require("model_path"),
        config,
    )
    trainer = NERTrainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        train_dataset=train_dataset,
        dev_dataset=dev_dataset,
        collator=collator,
    )

    try:
        summary = trainer.train()
        print(
            f"Training finished. Best epoch={summary['best_epoch']}, "
            f"dev F1={summary['best_f1']:.4f}"
        )
        if config.get("run_test_after_train", False):
            test_result = trainer.test(test_dataset)
            print(
                "Test result: "
                f"P={test_result['precision']:.4f}, "
                f"R={test_result['recall']:.4f}, "
                f"F1={test_result['f1']:.4f}"
            )
    finally:
        trainer.finish()


if __name__ == "__main__":
    main()
