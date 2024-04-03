# Provision an EC2 instance with NVIDIA T4 GPU in AWS Cape Town region using Terraform
# NOTE: You will need to install the NVIDIA drivers manually on the instance to enable GPU support for model training
provider "aws" {
  region = "af-south-1"  # Cape Town region
}

# Define variables
variable "instance_type" {
  description = "EC2 instance type"
  default     = "g4dn.xlarge"  # Instance type with NVIDIA T4 GPU
}

variable "ami_id" {
  description = "AMI ID for the EC2 instance"
  default     = "ami-0c1a7f89451184c8b"  # Ubuntu 20.04 LTS AMI ID for Cape Town region
}

variable "key_name" {
  description = "Name of the key pair"
  default     = "my-key-pair"
}

variable "public_key_path" {
  description = "Path to the local SSH public key file"
  default     = "~/.ssh/id_rsa.pub"
}

# Create a key pair using the local SSH public key
resource "aws_key_pair" "my_key_pair" {
  key_name   = var.key_name
  public_key = file(var.public_key_path)
}

# Create an EC2 instance
resource "aws_instance" "gpu_instance" {
  ami           = var.ami_id
  instance_type = var.instance_type
  key_name      = aws_key_pair.my_key_pair.key_name

  root_block_device {
    volume_size = 100  # 100GB disk storage
  }

  tags = {
    Name = "bezos-t4-001"
  }
}

# Output the public IP address of the EC2 instance
output "instance_public_ip" {
  value = aws_instance.gpu_instance.public_ip
}