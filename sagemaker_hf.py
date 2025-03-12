from sagemaker.huggingface import HuggingFace

estimator = HuggingFace(
    entry_point="train.py",
    role="arn:aws:iam::123456789012:role/SageMakerRole",
    instance_count=1,
    instance_type="ml.g5.4xlarge",
    transformers_version="4.28",
    pytorch_version="2.0",
    py_version="py310",
    hyperparameters={
        "num_train_epochs": 2,
        "per_device_train_batch_size": 1,
        "learning_rate": 1e-5,
    },
)

estimator.fit()

predictor = estimator.deploy(initial_instance_count=1, instance_type="ml.g5.2xlarge")
print("Model deployed at endpoint:", predictor.endpoint_name)
