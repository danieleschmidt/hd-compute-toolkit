# Global HD-Compute Infrastructure
terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
  }
}

# Multi-region deployment
module "us_west_2" {
  source = "./modules/hd-compute-cluster"
  
  region = "us-west-2"
  cluster_name = "hd-compute-us-west-2"
  node_count = 3
  instance_type = "c5.2xlarge"
  
  compliance_mode = "CCPA"
  supported_languages = ["en", "es"]
  
  tags = {
    Environment = "production"
    Region = "us-west-2"
    Compliance = "CCPA"
  }
}

module "eu_west_1" {
  source = "./modules/hd-compute-cluster"
  
  region = "eu-west-1"
  cluster_name = "hd-compute-eu-west-1"
  node_count = 2
  instance_type = "c5.xlarge"
  
  compliance_mode = "GDPR"
  supported_languages = ["en", "fr", "de"]
  
  tags = {
    Environment = "production"
    Region = "eu-west-1"
    Compliance = "GDPR"
  }
}

module "ap_southeast_1" {
  source = "./modules/hd-compute-cluster"
  
  region = "ap-southeast-1"
  cluster_name = "hd-compute-ap-southeast-1"
  node_count = 2
  instance_type = "c5.xlarge"
  
  compliance_mode = "PDPA"
  supported_languages = ["en", "zh", "ja"]
  
  tags = {
    Environment = "production"
    Region = "ap-southeast-1" 
    Compliance = "PDPA"
  }
}

# Global load balancer
resource "aws_globalaccelerator_accelerator" "hd_compute_global" {
  name            = "hd-compute-global"
  ip_address_type = "IPV4"
  enabled         = true

  attributes {
    flow_logs_enabled   = true
    flow_logs_s3_bucket = aws_s3_bucket.global_logs.bucket
    flow_logs_s3_prefix = "accelerator-logs/"
  }

  tags = {
    Name = "HD-Compute Global Accelerator"
  }
}

# S3 bucket for global logs
resource "aws_s3_bucket" "global_logs" {
  bucket = "hd-compute-global-logs"
  
  tags = {
    Name = "HD-Compute Global Logs"
  }
}

# CloudWatch dashboard for global monitoring
resource "aws_cloudwatch_dashboard" "global" {
  dashboard_name = "HD-Compute-Global"

  dashboard_body = jsonencode({
    widgets = [
      {
        type   = "metric"
        x      = 0
        y      = 0
        width  = 12
        height = 6

        properties = {
          metrics = [
            ["AWS/ApplicationELB", "RequestCount", "LoadBalancer", module.us_west_2.load_balancer_name],
            [".", ".", ".", module.eu_west_1.load_balancer_name],
            [".", ".", ".", module.ap_southeast_1.load_balancer_name]
          ]
          period = 300
          stat   = "Sum"
          region = "us-west-2"
          title  = "Global Request Count"
        }
      }
    ]
  })
}