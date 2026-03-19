# Terraform AWS S3 remote for DVC

This folder provisions an S3 bucket that can be used as a DVC remote.

## Usage

```bash
cd terraform/aws_dvc_remote
cp terraform.tfvars.example terraform.tfvars
terraform init
terraform plan
terraform apply
```

Then configure DVC in the project root:

```bash
dvc remote add -d s3remote s3://<your-bucket-name>
dvc remote modify s3remote region eu-west-3
```

If you authenticate with AWS environment variables:

```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_DEFAULT_REGION=eu-west-3
```
