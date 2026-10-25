"""Launch the VizWiz LoRA fine-tune as a SageMaker training job.

    pip install sagemaker
    python training/sagemaker/launch_training.py \
        --role-arn <SageMakerRoleArn stack output> \
        --data s3://<DataBucket>/datasets/vizwiz/ \
        --max-samples 4000

Default instance ml.g5.2xlarge (1x A10G 24 GB). Needs a service-quota for
"ml.g5.2xlarge for training job usage" >= 1 in your region (docs/AWS_SETUP.md).
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--role-arn", required=True)
    ap.add_argument("--data", required=True, help="s3:// prefix holding sft_train.jsonl + images/train/")
    ap.add_argument("--instance-type", default="ml.g5.2xlarge")
    ap.add_argument("--model-id", default="Qwen/Qwen2-VL-2B-Instruct")
    ap.add_argument("--max-samples", type=int, default=4000)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--transformers-version", default="4.49.0")
    ap.add_argument("--pytorch-version", default="2.5.1")
    ap.add_argument("--py-version", default="py311")
    ap.add_argument("--spot", action="store_true", help="managed spot training (~60-70%% cheaper, may be interrupted)")
    args = ap.parse_args()

    from sagemaker.huggingface import HuggingFace

    est = HuggingFace(
        entry_point="train_vizwiz_lora.py",
        source_dir=str(Path(__file__).parent),
        role=args.role_arn,
        instance_type=args.instance_type,
        instance_count=1,
        transformers_version=args.transformers_version,
        pytorch_version=args.pytorch_version,
        py_version=args.py_version,
        volume_size=200,
        max_run=6 * 3600,
        use_spot_instances=args.spot,
        max_wait=8 * 3600 if args.spot else None,
        hyperparameters={"model-id": args.model_id, "max-samples": args.max_samples, "epochs": args.epochs},
        base_job_name="uncane-vizwiz-lora",
        disable_profiler=True,
    )
    est.fit({"train": args.data}, wait=False)
    job = est.latest_training_job.name
    print(f"started training job {job}")
    print(f"  follow logs:  aws sagemaker describe-training-job --training-job-name {job} --query TrainingJobStatus")
    print("  when done, the model artifact S3 URI is in ModelArtifacts.S3ModelArtifacts -> deploy_endpoint.py")


if __name__ == "__main__":
    main()
