# Configure the AWS Provider
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# Provider configuration for us-west-2
provider "aws" {
  region = "us-west-2"
  alias  = "us_west"
}

# Provider configuration for eu-central-1
provider "aws" {
  region = "eu-central-1"
  alias  = "eu_central"
}

# Variables
variable "application_name" {
  type    = string
  default = "geo-routing-app"
}

variable "environment_name_us" {
  type    = string
  default = "us-west-env"
}

variable "environment_name_eu" {
  type    = string
  default = "eu-central-env"
}

variable "solution_stack_name" {
  type    = string
  default = "64bit Amazon Linux 2023 v4.0.0 running Python 3.11"
}

variable "instance_type" {
  type    = string
  default = "t3.micro"
}

variable "key_name" {
  type    = string
  default = "your_key_pair"
}

# Create an Elastic Beanstalk Application
resource "aws_elastic_beanstalk_application" "application" {
  name        = var.application_name
  description = "Elastic Beanstalk Application for Geo-Routing"
}

# IAM role for Elastic Beanstalk
resource "aws_iam_role" "aws_elasticbeanstalk_ec2_role" {
  name = "aws-elasticbeanstalk-ec2-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Action = "sts:AssumeRole",
        Principal = {
          Service = "ec2.amazonaws.com"
        },
        Effect = "Allow",
        Sid    = ""
      },
    ]
  })
}

resource "aws_iam_instance_profile" "aws_elasticbeanstalk_ec2_profile" {
  name = "aws-elasticbeanstalk-ec2-profile"
  role = aws_iam_role.aws_elasticbeanstalk_ec2_role.name
}

resource "aws_iam_role_policy_attachment" "aws_elasticbeanstalk_ec2_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AWSElasticBeanstalkWebTier"
  role       = aws_iam_role.aws_elasticbeanstalk_ec2_role.name
}

resource "aws_iam_role_policy_attachment" "aws_elasticbeanstalk_ec2_policy_2" {
  policy_arn = "arn:aws:iam::aws:policy/AWSElasticBeanstalkMulticontainerDocker"
  role       = aws_iam_role.aws_elasticbeanstalk_ec2_role.name
}

resource "aws_iam_role_policy_attachment" "aws_elasticbeanstalk_ec2_policy_3" {
  policy_arn = "arn:aws:iam::aws:policy/AWSElasticBeanstalkWorkerTier"
  role       = aws_iam_role.aws_elasticbeanstalk_ec2_role.name
}

# Elastic Beanstalk Environment in us-west-2
resource "aws_elastic_beanstalk_environment" "us_west" {
  name          = var.environment_name_us
  application   = aws_elastic_beanstalk_application.application.name
  solution_stack_name = var.solution_stack_name

  setting {
    namespace = "aws:autoscaling:launchconfiguration"
    name      = "InstanceType"
    value     = var.instance_type
  }

  setting {
    namespace = "aws:ec2:keypair"
    name      = "KeyName"
    value     = var.key_name
  }

  setting {
    namespace = "aws:elasticbeanstalk:environment"
    name      = "ServiceRole"
    value     = aws_iam_role.aws_elasticbeanstalk_ec2_role.name
  }

  setting {
    namespace = "aws:autoscaling:launchconfiguration"
    name      = "IamInstanceProfile"
    value     = aws_iam_instance_profile.aws_elasticbeanstalk_ec2_profile.name
  }

  provider = aws.us_west
}

# Elastic Beanstalk Environment in eu-central-1
resource "aws_elastic_beanstalk_environment" "eu_central" {
  name          = var.environment_name_eu
  application   = aws_elastic_beanstalk_application.application.name
  solution_stack_name = var.solution_stack_name

  setting {
    namespace = "aws:autoscaling:launchconfiguration"
    name      = "InstanceType"
    value     = var.instance_type
  }

  setting {
    namespace = "aws:ec2:keypair"
    name      = "KeyName"
    value     = var.key_name
  }

  setting {
    namespace = "aws:elasticbeanstalk:environment"
    name      = "ServiceRole"
    value     = aws_iam_role.aws_elasticbeanstalk_ec2_role.name
  }

  setting {
    namespace = "aws:autoscaling:launchconfiguration"
    name      = "IamInstanceProfile"
    value     = aws_iam_instance_profile.aws_elasticbeanstalk_ec2_profile.name
  }

  provider = aws.eu_central
}

# Route53 Zone (Replace with your actual domain)
resource "aws_route53_zone" "primary" {
  name = "example.com" # Replace with your domain
  comment = "Route53 Zone for Geo-Routing"
}

# Route53 Record for US West
resource "aws_route53_record" "us_west_record" {
  zone_id = aws_route53_zone.primary.zone_id
  name    = "geo.example.com" # Replace with your subdomain
  type    = "CNAME"
  ttl     = "300"

  set_identifier = "US"
  geolocation_routing_policy {
    country = "US"
  }

  records = [aws_elastic_beanstalk_environment.us_west.cname]
}

# Route53 Record for Europe Central
resource "aws_route53_record" "eu_central_record" {
  zone_id = aws_route53_zone.primary.zone_id
  name    = "geo.example.com" # Replace with your subdomain
  type    = "CNAME"
  ttl     = "300"

  set_identifier = "EU"
  geolocation_routing_policy {
    continent = "EU"
  }

  records = [aws_elastic_beanstalk_environment.eu_central.cname]
}

# Route53 Record for Default (if no geolocation matches)
resource "aws_route53_record" "default_record" {
  zone_id = aws_route53_zone.primary.zone_id
  name    = "geo.example.com" # Replace with your subdomain
  type    = "CNAME"
  ttl     = "300"

  set_identifier = "Default"
  geolocation_routing_policy {
    default = true
  }

  records = [aws_elastic_beanstalk_environment.us_west.cname] # Default to US West
}