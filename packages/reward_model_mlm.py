import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset, Dataset as HFDataset, DatasetDict
from peft import LoraConfig, get_peft_model, TaskType
from torch.optim import AdamW
import torch.nn.functional as F
from typing import Dict, Tuple, Any, Union


class PreferenceDataset(Dataset):
    """
    Custom PyTorch Dataset for preference-based reward modeling.
    Takes a Hugging Face dataset with 'prompt', 'chosen', and 'rejected' columns.
    """
    def __init__(self, hf_dataset: HFDataset, tokenizer: Any, max_length: int = 512):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def _format_and_tokenize(self, prompt: str, response: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Helper function to format and tokenize a prompt-response pair."""
        conversation = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
        formatted_text = self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
        tokenized = self.tokenizer(
            formatted_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        # Remove the batch dimension added by the tokenizer
        return tokenized["input_ids"].squeeze(0), tokenized["attention_mask"].squeeze(0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.hf_dataset[idx]
        prompt = item["prompt"]
        chosen = item["chosen"]
        rejected = item["rejected"]

        chosen_input_ids, chosen_attention_mask = self._format_and_tokenize(prompt, chosen)
        rejected_input_ids, rejected_attention_mask = self._format_and_tokenize(prompt, rejected)

        return {
            "chosen_input_ids": chosen_input_ids,
            "chosen_attention_mask": chosen_attention_mask,
            "rejected_input_ids": rejected_input_ids,
            "rejected_attention_mask": rejected_attention_mask,
        }


class PreferenceDataModule(pl.LightningDataModule):
    """
    Handles loading the preference dataset, creating DataLoaders, and managing tokenization.
    """
    def __init__(
            self,
            model_name: str,
            dataset: Union[Dict[str, HFDataset], DatasetDict],
            batch_size: int = 8,
            max_length: int = 512,
            train_split: str = "train",
            val_split: str = "validation"
        ):
        super().__init__()
        self.model_name = model_name
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_length = max_length
        self.train_split = train_split
        self.val_split = val_split
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Some models don't have a pad token, so we set it to the EOS token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def setup(self, stage=None) -> None:
        # Load dataset from Hugging Face (or a local path)
        self.train_dataset = PreferenceDataset(self.dataset[self.train_split], self.tokenizer, self.max_length)
        self.val_dataset = PreferenceDataset(self.dataset[self.val_split], self.tokenizer, self.max_length)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4, pin_memory=True)

# 3. PyTorch LightningModule for the Preference Model
class PreferenceModel(pl.LightningModule):
    """
    The core LightningModule that defines the model and Bradley-Terry loss.
    """
    def __init__(self,
            model_name: str,
            learning_rate: float = 2e-5,
        ):
        super().__init__()
        self.learning_rate = learning_rate
        self.save_hyperparameters()

        # Setup to store score range
        self.lowest_logits = None
        self.highest_logits = None

        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1,
            dtype=torch.bfloat16
        )
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1
        )
        self.model = get_peft_model(base_model, peft_config)
        self.model.print_trainable_parameters()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits.squeeze(-1)

    def _common_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Tuple[torch.Tensor, float, float, float, float]:
        rewards_chosen = self(input_ids=batch["chosen_input_ids"], attention_mask=batch["chosen_attention_mask"])
        rewards_rejected = self(input_ids=batch["rejected_input_ids"], attention_mask=batch["rejected_attention_mask"])

        # Bradley-Terry loss
        loss = -F.logsigmoid(rewards_chosen - rewards_rejected).mean()

        # Metrics
        accuracy = (rewards_chosen > rewards_rejected).float().mean()
        MAE = torch.abs(rewards_chosen - rewards_rejected).mean()
        lowest_logits = torch.min(rewards_rejected)
        highest_logits = torch.max(rewards_chosen)

        return loss, accuracy, MAE, lowest_logits, highest_logits

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss, accuracy, MAE, lowest_logits, highest_logits = self._common_step(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log("train_accuracy", accuracy, prog_bar=True, on_step=True, on_epoch=False)
        self.log("train_MAE", MAE, prog_bar=True, on_step=True, on_epoch=False)
        
        # Track range
        if not self.lowest_logits or lowest_logits < self.lowest_logits:
            self.lowest_logits = lowest_logits
            self.log("train_lowest_logits", lowest_logits, prog_bar=True, on_step=False, on_epoch=True)
        if not self.highest_logits or highest_logits > self.highest_logits:
            self.highest_logits = highest_logits
            self.log("train_highest_logits", highest_logits, prog_bar=True, on_step=False, on_epoch=True)
    
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss, accuracy, MAE, _, _ = self._common_step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_accuracy", accuracy, prog_bar=True, on_epoch=True)
        self.log("val_MAE", MAE, prog_bar=True, on_epoch=True)

        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        # scheduler = get_linear_schedule_with_warmup(
        #     optimizer,
        #     num_warmup_steps=self.hparams.warmup_steps,
        #     num_training_steps=self.hparams.total_steps
        # )
        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}
        # }
        return optimizer

# 4. Main Training Function
def main() -> None:
    # --- Configuration ---
    MODEL_NAME = "Skywork/Skywork-Reward-V2-Qwen3-0.6B" # A smaller model for a quick demonstration
    
    dummy_data = {
        "train": HFDataset.from_dict({
            "prompt": ["What is 2+2?", "What is the capital of France?"],
            "chosen": ["2+2 is 4.", "Paris is the capital of France."],
            "rejected": ["2+2 is 5.", "The capital of France is Berlin."]
        }),
        "validation": HFDataset.from_dict({
            "prompt": ["Explain gravity.", "Who wrote Hamlet?"],
            "chosen": ["Gravity is a fundamental interaction of nature that causes mutual attraction between all things with mass or energy.", "William Shakespeare wrote Hamlet."],
            "rejected": ["Gravity is a suggestion.", "Hamlet was written by Leonardo da Vinci."]
        })
    }
    # score_chosen, score_rejected in dataset item

    # DUMMY_DATASET_PATH = "../dummy_preference_dataset"
    # DatasetDict(dummy_data).save_to_disk(DUMMY_DATASET_PATH)

    BATCH_SIZE = 2
    MAX_LENGTH = 512
    LEARNING_RATE = 2e-5
    EPOCHS = 3

    # --- Setup ---
    pl.seed_everything(42)

    data_module = PreferenceDataModule(
        model_name=MODEL_NAME,
        dataset=dummy_data,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH
    )
    data_module.setup()

    preference_model = PreferenceModel(
        model_name=MODEL_NAME,
        learning_rate=LEARNING_RATE,
    )

    # --- Training ---
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="auto", # Automatically selects GPU if available
        devices=1,
        # precision="16-mixed" # Use mixed precision for better performance and less memory
    )

    trainer.fit(preference_model, data_module)

    # --- Save the final model ---
    # final_model_path = "../my_preference_model"
    # preference_model.model.save_pretrained(final_model_path)
    # data_module.tokenizer.save_pretrained(final_model_path)
    # print(f"\nTraining complete! Model saved to {final_model_path}")

if __name__ == "__main__":
    main()
