# Provision a GPU-enabled Compute Engine instance on Google Cloud Platform (GCP) using Terraform
# NOTE: You will need to install the NVIDIA drivers manually on the instance to enable GPU support for model training
provider "google" {
  project = "zabantu-project-01"
  region  = "europe-west4-b"
}

# Define variables
variable "instance_name" {
  description = "Name of the Compute Engine instance"
  default     = "gpu-instance-0001"
}

variable "machine_type" {
  description = "Machine type for the Compute Engine instance"
  default     = "n1-standard-4"
}

variable "gpu_type" {
  description = "Type of GPU to attach to the instance"
  default     = "nvidia-tesla-t4"
}

variable "gpu_count" {
  description = "Number of GPUs to attach to the instance"
  default     = 1
}

variable "boot_disk_size" {
  description = "Size of the boot disk in GB"
  default     = 100
}

variable "ssh_public_key_path" {
  description = "Path to the local SSH public key file"
  default     = "~/.ssh/id_rsa.pub"
}

# Create a Compute Engine instance
resource "google_compute_instance" "gpu_instance" {
  name         = var.instance_name
  machine_type = var.machine_type
  zone         = "africa-south1-a"

  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-2004-lts"
      size  = var.boot_disk_size
    }
  }

  guest_accelerator {
    type  = var.gpu_type
    count = var.gpu_count
  }

  network_interface {
    network = "default"
    access_config {}
  }

  metadata = {
    ssh-keys = "ubuntu:${file(var.ssh_public_key_path)}"
  }

  tags = ["gpu-instance"]
}

# Output the public IP address of the Compute Engine instance
output "gcp_instance_public_ip" {
  value = google_compute_instance.gpu_instance.network_interface[0].access_config[0].nat_ip
}