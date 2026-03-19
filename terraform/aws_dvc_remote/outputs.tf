output "bucket_name" {
  description = "S3 bucket name for DVC remote storage."
  value       = aws_s3_bucket.dvc.bucket
}

output "bucket_arn" {
  description = "S3 bucket ARN."
  value       = aws_s3_bucket.dvc.arn
}

output "dvc_remote_url" {
  description = "Value to use with 'dvc remote add'."
  value       = "s3://${aws_s3_bucket.dvc.bucket}"
}
