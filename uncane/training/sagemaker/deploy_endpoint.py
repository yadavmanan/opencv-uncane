"""Deploy the fine-tuned VLM to a real-time SageMaker endpoint (Hugging Face TGI).

    python training/sagemaker/deploy_endpoint.py \
        --role-arn <SageMakerRoleArn> \
        --model-data s3://.../output/model.tar.gz \
        --endpoint-name uncane-vizwiz-vlm

Then point the Lambda at it:
    sam deploy --parameter-overrides VlmBackend=sagemaker SageMakerEndpointName=uncane-vizwiz-vlm

A g5.xlarge endpoint bills every hour it exists. Delete it when not demoing:
    python training/sagemaker/deploy_endpoint.py --delete --endpoint-name uncane-vizwiz-vlm
"""
from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--role-arn")
    ap.add_argument("--model-data", help="s3://.../model.tar.gz from the training job")
    ap.add_argument("--endpoint-name", default="uncane-vizwiz-vlm")
    ap.add_argument("--instance-type", default="ml.g5.xlarge")
    ap.add_argument("--image-uri", help="override the TGI container image")
    ap.add_argument("--delete", action="store_true")
    args = ap.parse_args()

    import boto3

    if args.delete:
        sm = boto3.client("sagemaker")
        sm.delete_endpoint(EndpointName=args.endpoint_name)
        print(f"deleted endpoint {args.endpoint_name} (the endpoint config and model remain; delete in console if unused)")
        return
    if not (args.role_arn and args.model_data):
        ap.error("--role-arn and --model-data are required to deploy")

    from sagemaker.huggingface import HuggingFaceModel, get_huggingface_llm_image_uri

    image = args.image_uri or get_huggingface_llm_image_uri("huggingface")
    model = HuggingFaceModel(
        model_data=args.model_data,
        role=args.role_arn,
        image_uri=image,
        env={
            "HF_MODEL_ID": "/opt/ml/model",
            "SM_NUM_GPUS": "1",
            "MAX_INPUT_TOKENS": "3072",
            "MAX_TOTAL_TOKENS": "4096",
            "MAX_BATCH_PREFILL_TOKENS": "4096",
        },
    )
    model.deploy(initial_instance_count=1, instance_type=args.instance_type, endpoint_name=args.endpoint_name,
                 container_startup_health_check_timeout=900)
    print(f"endpoint {args.endpoint_name} is InService (image {image})")


if __name__ == "__main__":
    main()
