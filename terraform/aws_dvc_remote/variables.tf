variable "aws_region" {
  description = "AWS region used for the DVC remote bucket."
  type        = string
  default     = "eu-west-3"
}

variable "bucket_name" {
  description = "Globally unique S3 bucket name for DVC artifacts."
  type        = string
}

variable "enable_versioning" {
  description = "Enable S3 versioning for the DVC bucket."
  type        = bool
  default     = true
}

variable "tags" {
  description = "Extra tags applied to the bucket."
  type        = map(string)
  default = {
    ManagedBy = "Terraform"
  }
}
