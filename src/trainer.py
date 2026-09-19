import gc
import math
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from src.evaluator import evaluate_dataset
from src.utils import gpu_memory_gb, model_device, save_json, set_seed


class NERTrainer:
    def __init__(
        self,
        model,
        tokenizer,
        config,
        train_dataset,
        dev_dataset,
        collator,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.collator = collator
        self.output_dir = Path(config.require("output_dir"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        save_json(self.output_dir / "config.json", config.values)

        set_seed(config.get("seed", 42))
        self.device = model_device(model)
        self.trainable_params = [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad # 只有LoRA的A/B矩阵可训练
        ]
        self.optimizer = AdamW(
            self.trainable_params,
            lr=config.get("learning_rate", 2e-5),
            weight_decay=config.get("weight_decay", 0.01),
        )

        steps_per_epoch = math.ceil(
            len(train_dataset) / config.get("batch_size", 4)
        )
        accumulation_steps = config.get("gradient_accumulation_steps", 1)
        self.total_steps = (
            math.ceil(steps_per_epoch / accumulation_steps)
            * config.get("num_epochs", 5)
        )
        warmup_steps = int(
            self.total_steps * config.get("warmup_ratio", 0.03)
        )
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=self.total_steps,
        )

        self.best_f1 = -1.0
        self.best_epoch = -1
        self.best_result = None
        self.no_improvement = 0
        self.global_step = 0
        self.history = []
        self.swanlab = None

    def _init_swanlab(self):
        if not self.config.get("use_swanlab", False):
            return
        try:
            import swanlab
        except ImportError:
            print("SwanLab is not installed; training will continue without it.")
            return

        try:
            self.swanlab = swanlab.init(
                project=self.config.get(
                    "swanlab_project",
                    "demo3-qwen7b-bc2gm",
                ),
                experiment_name=self.config.require("experiment_name"),
                mode=self.config.get("swanlab_mode", "cloud"),
                config=self.config.values,
            )
        except Exception as error:
            print(f"SwanLab initialization failed: {error}")
            self.swanlab = None

    def _log(self, values, step=None):
        if self.swanlab is None:
            return
        if step is None:
            self.swanlab.log(values)
        else:
            self.swanlab.log(values, step=step)

    def _save_adapter(self, name):
        path = self.output_dir / name
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        return path

    def _save_loss_plot(self):
        if not self.history:
            return
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return

        epochs = [item["epoch"] for item in self.history]
        losses = [item["train_loss"] for item in self.history]
        plt.figure(figsize=(7, 4))
        plt.plot(epochs, losses, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Train Loss")
        plt.title("Training Loss")
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(self.output_dir / "training_loss.png", dpi=160)
        plt.close()

    def _train_loader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.get("batch_size", 4),
            shuffle=True,
            collate_fn=self.collator,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

    def _move_batch(self, batch):
        return {
            key: value.to(self.device, non_blocking=True)
            for key, value in batch.items()
        }

    def train(self):
        self._init_swanlab()
        self.model.train()
        self.model.config.use_cache = False
        train_loader = self._train_loader()
        accumulation_steps = self.config.get(
            "gradient_accumulation_steps",
            1,
        )
        num_epochs = self.config.get("num_epochs", 5)
        max_grad_norm = self.config.get("max_grad_norm", 1.0)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

        start_time = time.time()
        self.optimizer.zero_grad(set_to_none=True)

        for epoch in range(num_epochs):
            epoch_start = time.time()
            epoch_loss = 0.0
            batch_count = 0
            progress = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{num_epochs}",
            )

            for step, batch in enumerate(progress):
                batch = self._move_batch(batch)
                autocast_enabled = (
                    torch.cuda.is_available()
                    and self.config.get("bf16", True)
                )
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=autocast_enabled,
                ):
                    outputs = self.model(
                        **batch,
                        use_cache=False,
                    )
                    loss = outputs.loss

                is_last_batch = step + 1 == len(train_loader)
                current_accumulation = accumulation_steps
                if is_last_batch and len(train_loader) % accumulation_steps:
                    current_accumulation = len(train_loader) % accumulation_steps
                (loss / current_accumulation).backward()
                epoch_loss += loss.detach().float().item()
                batch_count += 1

                should_step = (
                    (step + 1) % accumulation_steps == 0
                    or is_last_batch
                ) # 是否需要执行参数更新
                if should_step:
                    torch.nn.utils.clip_grad_norm_(
                        self.trainable_params,
                        max_grad_norm,
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1
                    progress.set_postfix(
                        loss=f"{loss.detach().float().item():.4f}",
                        lr=f"{self.scheduler.get_last_lr()[0]:.2e}",
                    )
                    self._log(
                        {
                            "train/loss": loss.detach().float().item(),
                            "train/learning_rate": self.scheduler.get_last_lr()[0],
                        },
                        step=self.global_step,
                    )

                del batch, outputs, loss

            train_loss = epoch_loss / max(batch_count, 1)
            eval_result = evaluate_dataset(
                model=self.model,
                tokenizer=self.tokenizer,
                dataset=self.dev_dataset,
                instruction=self.config.require("instruction"),
                batch_size=self.config.get("eval_batch_size", 4),
                max_length=self.config.get("max_length", 256),
                max_new_tokens=self.config.get("max_new_tokens", 256),
                description=f"Dev epoch {epoch + 1}",
            )

            if torch.cuda.is_available():
                peak_gb = gpu_memory_gb()
            else:
                peak_gb = 0.0

            epoch_result = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "precision": eval_result["precision"],
                "recall": eval_result["recall"],
                "f1": eval_result["f1"],
                "tp": eval_result["tp"],
                "fp": eval_result["fp"],
                "fn": eval_result["fn"],
                "peak_gpu_gb": peak_gb,
                "epoch_seconds": time.time() - epoch_start,
            }
            self.history.append(epoch_result)
            print(
                f"Epoch {epoch + 1}: loss={train_loss:.4f} "
                f"P={eval_result['precision']:.4f} "
                f"R={eval_result['recall']:.4f} "
                f"F1={eval_result['f1']:.4f} "
                f"peak_gpu={peak_gb:.2f}GB"
            )
            self._log(
                {
                    "eval/precision": eval_result["precision"],
                    "eval/recall": eval_result["recall"],
                    "eval/f1": eval_result["f1"],
                    "eval/peak_gpu_gb": peak_gb,
                    "train/epoch_loss": train_loss,
                },
                step=self.global_step,
            )

            if eval_result["f1"] > self.best_f1:
                self.best_f1 = eval_result["f1"]
                self.best_epoch = epoch + 1
                self.best_result = eval_result.copy()
                self.no_improvement = 0
                self._save_adapter("best")
                print(f"Saved new best adapter: {self.output_dir / 'best'}")
            else:
                self.no_improvement += 1

            self._save_adapter("last")
            save_json(
                self.output_dir / "history.json",
                {
                    "history": self.history,
                    "best_epoch": self.best_epoch,
                    "best_f1": self.best_f1,
                },
            )

            if self.no_improvement >= self.config.get(
                "early_stopping_patience",
                2,
            ):
                print("Early stopping triggered.")
                break

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self._save_loss_plot()
        total_seconds = time.time() - start_time
        summary = {
            "best_epoch": self.best_epoch,
            "best_f1": self.best_f1,
            "best_result": self.best_result,
            "train_seconds": total_seconds,
            "history": self.history,
        }
        save_json(self.output_dir / "train_results.json", summary)
        self._log(
            {
                "final/best_f1": self.best_f1,
                "final/best_epoch": self.best_epoch,
                "final/train_seconds": total_seconds,
            }
        )
        return summary

    def test(self, test_dataset):
        best_path = self.output_dir / "best"
        if not best_path.exists():
            raise FileNotFoundError(f"Best adapter not found: {best_path}")

        self.model.load_adapter(
            str(best_path),
            adapter_name="best",
            is_trainable=False,
        )
        self.model.set_adapter("best")
        result = evaluate_dataset(
            model=self.model,
            tokenizer=self.tokenizer,
            dataset=test_dataset,
            instruction=self.config.require("instruction"),
            batch_size=self.config.get("eval_batch_size", 4),
            max_length=self.config.get("max_length", 256),
            max_new_tokens=self.config.get("max_new_tokens", 256),
            limit=self.config.get("test_limit"),
            description="Test",
        )
        save_json(self.output_dir / "eval_test.json", result)
        return result

    def finish(self):
        if self.swanlab is not None:
            self.swanlab.finish()
