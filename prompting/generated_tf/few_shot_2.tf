terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = "us-east-1"
}

resource "aws_redshift_cluster" "default" {
  cluster_identifier      = "redshift-cluster"
  database_name           = "mydb"
  master_username         = "admin"
  master_password         = "SuperSecretPassword123"
  node_type               = "dc2.large"
  cluster_type            = "single-node"
  port                    = 5439
  automated_snapshot_retention_period = 1

  snapshot_schedule_arn = aws_redshift_snapshot_schedule.hourly.arn
}

resource "aws_redshift_snapshot_schedule" "hourly" {
  identifier = "hourly-snapshot-schedule"

  schedule_definitions = ["cron(0 0/12 * * ? *)"]
}