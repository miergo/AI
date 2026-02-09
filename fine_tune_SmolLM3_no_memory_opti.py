from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import trackio as wandb
import dotenv

dotenv.load_dotenv()

# Initialize experiment tracking
wandb.init(project="smollm3-sft", name="my-first-sft-run")

# Load SmolLM3 base model
model = AutoModelForCausalLM.from_pretrained(
    "HuggingFaceTB/SmolLM3-3B-Base", device_map={"": 0}
)
tokenizer = AutoTokenizer.from_pretrained(
    "HuggingFaceTB/SmolLM3-3B-Base", device_map={"": 0}
)

# Load SmolTalk2 dataset
dataset = load_dataset("HuggingFaceTB/smoltalk2_everyday_convs_think")

# Configure training with Trackio integration
config = SFTConfig(
    output_dir="./smollm3-finetuned",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=5e-5,
    max_steps=1000,
    report_to="trackio",  # Enable Trackio logging
)

# Train!
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    args=config,
)
trainer.train()
