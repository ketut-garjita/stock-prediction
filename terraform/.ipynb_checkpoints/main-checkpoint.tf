provider "aws" {
  region  = var.aws_region
  profile = "default"
}

# Elastic IP for stable public IP
resource "aws_eip" "stock_server_eip" {
  instance = aws_instance.stock_predict_server.id
}

# Security Group for EC2 instance
resource "aws_security_group" "allow_http_ssh" {
  name        = "allow_http_ssh_SG"
  description = "Allow HTTP and SSH access with restricted IPs"

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.ip_cidr_blocks]
  }

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["43.218.193.64/29"] # EC2 Instance Connect service IP range
  }

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"] # Allow HTTP access for all
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"] # Allow all outbound traffic
  }
}

# EC2 Instance
resource "aws_instance" "stock_predict_server" {
  ami           = var.ami_name
  instance_type = var.instance_type
  key_name      = var.key_name
  security_groups = [aws_security_group.allow_http_ssh.name]
  associate_public_ip_address = true

  root_block_device {
    volume_size = 20 # Ukuran disk dalam GB
    volume_type = "gp3" # Tipe volume EBS (default: General Purpose SSD)
  }
  
  user_data = <<-EOF
  #!/bin/bash
  exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1
  sudo apt update -y
  sudo apt install -y docker.io awscli unattended-upgrades
  sudo dpkg-reconfigure --priority=low unattended-upgrades
  sudo systemctl start docker
  sudo systemctl enable docker
  sudo usermod -aG docker $USER
  EOF

  tags = {
    Name        = var.ec2_instance_name
    Environment = "Development"
    Owner       = "Engineer"
  }
}

resource "null_resource" "copy_files" {
  provisioner "file" {
    source      = "/home/hduser/ml-zoomcamp/stock-prediction"
    destination = "/home/ubuntu/stock-prediction"

    connection {
      type        = "ssh"
      user        = "ubuntu"
      private_key = file("~/.ssh/stock-keypair.pem")
      host        = aws_instance.stock_predict_server.public_ip
    }
  }

  provisioner "remote-exec" {
    inline = [
      "ls -lah /home/ubuntu/stock-prediction",
    ]

    connection {
      type        = "ssh"
      user        = "ubuntu"
      private_key = file("~/.ssh/stock-keypair.pem")
      host        = aws_instance.stock_predict_server.public_ip
    }
  }

  depends_on = [aws_instance.stock_predict_server]
}


# Outputs
output "instance_public_ip" {
  value = aws_eip.stock_server_eip.public_ip
}

output "instance_id" {
  value = aws_instance.stock_predict_server.id
}

output "security_group_id" {
  value = aws_security_group.allow_http_ssh.id
}

output "elastic_ip" {
  value = aws_eip.stock_server_eip.public_ip
}
