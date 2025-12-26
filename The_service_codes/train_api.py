import subprocess
import os
import json
import time
from datetime import datetime

def print_status(msg):
    """Print status with timestamp"""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def main():
    # Core configuration
    llama_factory_dir = "/root/LLaMA-Factory"
    model_path = "/root/autodl-tmp/Qwen3-14B"
    dataset_name = "fine_tune_dataset_instruction"
    dataset_dir = f"{llama_factory_dir}/data"
    output_dir = "/root/autodl-tmp/qwen3-movie-results"
    template = "qwen3_nothink"
    log_file = f"{output_dir}/train_log_{datetime.now().strftime('%Y%m%d%H%M%S')}.txt"

    # Create directories (including log directory)
    os.makedirs(output_dir, exist_ok=True)
    print_status(f"Training logs will be saved to: {log_file}")

    # 1. Generate and validate dataset configuration
    print_status("=== Validate Dataset Configuration ===")
    dataset_info = {
        dataset_name: {
            "file_name": f"{dataset_name}.json",
            "formatting": "alpaca",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output"
            }
        }
    }
    # Write configuration
    config_path = f"{dataset_dir}/dataset_info.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)
    # Verify configuration file exists
    assert os.path.exists(config_path), f"Configuration file creation failed: {config_path}"
    # Verify dataset exists
    data_path = f"{dataset_dir}/{dataset_name}.json"
    assert os.path.exists(data_path), f"Dataset does not exist: {data_path}"
    print_status(f"✅ Dataset configuration validated: {data_path}")

    # 2. Training command (with validation, progress tracking, logging)
    print_status("=== Start Training ===")
    train_command = [
        "python", f"{llama_factory_dir}/src/train.py",
        "--stage", "sft",
        "--model_name_or_path", model_path,
        "--do_train", "True",
        "--do_eval", "True", 
        "--dataset", dataset_name,
        "--dataset_dir", dataset_dir,
        "--finetuning_type", "lora",
        "--quantization_bit", "4",
        "--per_device_train_batch_size", "2",
        "--gradient_accumulation_steps", "2",
        "--lora_rank", "16",
        "--gradient_checkpointing", "True",
        "--num_train_epochs", "4",
        "--learning_rate", "3e-5",
        "--save_steps", "500",
        "--eval_steps", "500", 
        "--logging_steps", "10",  
        "--output_dir", output_dir,
        "--overwrite_output_dir", "True",
        "--fp16", "True",
        "--val_size", "0.1",
        "--template", template,
        "--disable_tqdm", "False",  
        "--save_total_limit", "3"  
    ]

    # Set memory optimization environment variables
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # 3. Start training (write to log file simultaneously)
    with open(log_file, "a", encoding="utf-8") as f_log:
        process = subprocess.Popen(
            train_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        # Real-time printing + write to log
        for line in process.stdout:
            line_strip = line.strip()
            print(line_strip)
            f_log.write(line_strip + "\n")
            f_log.flush()  # Flush to write in real-time

        # Wait for training completion
        process.wait()

    # 4. Training result validation
    print_status("\n=== Validate Training Results ===")
    # Check if checkpoints are generated
    checkpoint_exists = any([f.startswith("checkpoint-") for f in os.listdir(output_dir)])
    # Check if evaluation results exist
    eval_result_exists = os.path.exists(f"{output_dir}/eval_results.json")

    if process.returncode == 0 and checkpoint_exists:
        print_status("✅ Training completed successfully!")
        print_status(f"📁 Training results directory: {output_dir}")
        if eval_result_exists:
            with open(f"{output_dir}/eval_results.json", "r") as f:
                eval_res = json.load(f)
            print_status(f"📊 Validation loss: {eval_res.get('eval_loss', 'N/A'):.4f}")
    else:
        print_status("❌ Training failed!")
        print_status(f"Log file: {log_file}")

if __name__ == "__main__":
    main()