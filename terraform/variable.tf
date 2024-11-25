## AWS account level config: region
variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "<your aws_region>"
}

## Key to allow connection to our EC2 instance
variable "key_name" {
  description = "EC2 key name"
  type        = string
  default     = "<your key_name>"
}

## EC2 instance type
variable "instance_type" {
  description = "Instance type for EMR and EC2"
  type        = string
  default     = "<your instance_type"
}

## EC2 instance name
variable "ec2_instance_name" {
  description = "Instance name for EMR and EC2"
  type        = string
  default     = "<your ec2_instance_name"
}

## cidr_blocks IP
# default: 0.0.0.0/0"
variable "ip_cidr_blocks" {
  description = "Security Group last name"
  type        = string
  default     = "<your ip_cidr_blocks>" 
}

## AMI
variable "ami_name" {
  description = "Security Group last name"
  type        = string
  default     = "<your ami_name>"
}
