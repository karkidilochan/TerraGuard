# Configure the AWS Provider
provider "aws" {
  region = "us-west-2"
  alias  = "us_west"
}

provider "aws" {
  region = "eu-central-1"
  alias  = "eu_central"
}

# Data source to get the current AWS caller identity
data "aws_caller_identity" "current" {}

# Create a Route53 zone
resource "aws_route53_zone" "primary" {
  name = "example.com" # Replace with your domain
}

# Elastic Beanstalk Application
resource "aws_elastic_beanstalk_application" "app" {
  name        = "my-app"
  description = "My Elastic Beanstalk Application"
}

# Elastic Beanstalk Environment in us-west-2
resource "aws_elastic_beanstalk_environment" "us_west" {
  name                = "us-west-env"
  application         = aws_elastic_beanstalk_application.app.name
  solution_stack_name = "64bit Amazon Linux 2023 v4.0.0 running Python 3.9" # Replace with a valid solution stack
  setting {
    namespace = "aws:autoscaling:launchconfiguration"
    name      = "InstanceType"
    value     = "t2.micro"
  }
  opts = {
    provider = aws.us_west
  }
}

# Elastic Beanstalk Environment in eu-central-1
resource "aws_elastic_beanstalk_environment" "eu_central" {
  name                = "eu-central-env"
  application         = aws_elastic_beanstalk_application.app.name
  solution_stack_name = "64bit Amazon Linux 2023 v4.0.0 running Python 3.9" # Replace with a valid solution stack
  setting {
    namespace = "aws:autoscaling:launchconfiguration"
    name      = "InstanceType"
    value     = "t2.micro"
  }
  opts = {
    provider = aws.eu_central
  }
}

# Route53 Record for us-west-2
resource "aws_route53_record" "us_west" {
  zone_id = aws_route53_zone.primary.zone_id
  name    = "www.example.com" # Replace with your subdomain
  type    = "CNAME"
  ttl     = "300"
  records = [aws_elastic_beanstalk_environment.us_west.cname]

  geolocation_routing_policy {
    country = "US" # Route US traffic to us-west-2
  }
}

# Route53 Record for eu-central-1
resource "aws_route53_record" "eu_central" {
  zone_id = aws_route53_zone.primary.zone_id
  name    = "www.example.com" # Replace with your subdomain
  type    = "CNAME"
  ttl     = "300"
  records = [aws_elastic_beanstalk_environment.eu_central.cname]

  geolocation_routing_policy {
    country = "DE" # Route German traffic to eu-central-1
  }
}