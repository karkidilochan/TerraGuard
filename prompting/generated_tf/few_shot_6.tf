# Configure the AWS Provider
provider "aws" {
  region = var.aws_region_us_west
  alias  = "us_west"
}

provider "aws" {
  region = var.aws_region_eu_central
  alias  = "eu_central"
}

# Variables
variable "aws_region_us_west" {
  type    = string
  default = "us-west-2"
}

variable "aws_region_eu_central" {
  type    = string
  default = "eu-central-1"
}

variable "application_name" {
  type    = string
  default = "geo-elastic-beanstalk"
}

variable "environment_name_us_west" {
  type    = string
  default = "us_west"
}

variable "environment_name_eu_central" {
  type    = string
  default = "eu_central"
}

variable "solution_stack_name" {
  type    = string
  default = "64bit Amazon Linux 2023 v4.0.0 running Python 3.11"
}

variable "instance_type" {
  type    = string
  default = "t3.micro"
}

# Elastic Beanstalk Application
resource "aws_elastic_beanstalk_application" "application" {
  name        = var.application_name
  description = "Elastic Beanstalk Application for Geo-Routing"
}

# IAM Role for Elastic Beanstalk
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
        Sid = ""
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "aws_elasticbeanstalk_ec2_role" {
  policy_arn = "arn:aws:iam::aws:policy/AWSElasticBeanstalkWebTier"
  role       = aws_iam_role.aws_elasticbeanstalk_ec2_role.name
}

resource "aws_iam_instance_profile" "aws_elasticbeanstalk_ec2_profile" {
  name = "aws-elasticbeanstalk-ec2-profile"
  role = aws_iam_role.aws_elasticbeanstalk_ec2_role.name
}

# Elastic Beanstalk Environment in us-west-2
resource "aws_elastic_beanstalk_environment" "us_west" {
  name        = var.environment_name_us_west
  application = aws_elastic_beanstalk_application.application.name
  solution_stack_name = var.solution_stack_name

  setting {
    namespace = "aws:autoscaling:launchconfiguration"
    name      = "InstanceType"
    value     = var.instance_type
  }

  setting {
    namespace = "aws:autoscaling:launchconfiguration"
    name      = "IamInstanceProfile"
    value     = aws_iam_instance_profile.aws_elasticbeanstalk_ec2_profile.name
  }

  opts = {
    provider = aws.us_west
  }
}

# Elastic Beanstalk Environment in eu-central-1
resource "aws_elastic_beanstalk_environment" "eu_central" {
  name        = var.environment_name_eu_central
  application = aws_elastic_beanstalk_application.application.name
  solution_stack_name = var.solution_stack_name

  setting {
    namespace = "aws:autoscaling:launchconfiguration"
    name      = "InstanceType"
    value     = var.instance_type
  }

  setting {
    namespace = "aws:autoscaling:launchconfiguration"
    name      = "IamInstanceProfile"
    value     = aws_iam_instance_profile.aws_elasticbeanstalk_ec2_profile.name
  }

  opts = {
    provider = aws.eu_central
  }
}

# Route53 Geolocation Routing
resource "aws_route53_zone" "primary" {
  name = "example.com" # Replace with your domain
}

resource "aws_route53_record" "us_west" {
  zone_id = aws_route53_zone.primary.zone_id
  name    = "example.com" # Replace with your domain
  type    = "CNAME"
  ttl     = "300"
  records = [aws_elastic_beanstalk_environment.us_west.endpoint_url]

  geolocation_routing_policy {
    country = "US"
  }
}

resource "aws_route53_record" "eu_central" {
  zone_id = aws_route53_zone.primary.zone_id
  name    = "example.com" # Replace with your domain
  type    = "CNAME"
  ttl     = "300"
  records = [aws_elastic_beanstalk_environment.eu_central.endpoint_url]

  geolocation_routing_policy {
    country = "DE"
  }
}

resource "aws_route53_record" "default" {
  zone_id = aws_route53_zone.primary.zone_id
  name    = "example.com" # Replace with your domain
  type    = "CNAME"
  ttl     = "300"
  records = [aws_elastic_beanstalk_environment.us_west.endpoint_url] # Default to us-west

  geolocation_routing_policy {
    default = true
  }
}