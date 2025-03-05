# Configure the AWS Provider
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

# Create a Redshift Cluster
resource "aws_redshift_cluster" "default" {
  cluster_identifier      = "terra-redshift-cluster"
  database_name           = "mydb"
  master_username         = "admin"
  master_password         = "Password123!"
  node_type               = "dc2.large"
  cluster_type            = "single-node"
  port                    = 5439
  skip_final_snapshot     = true
}

# Create a Redshift Snapshot Schedule
resource "aws_redshift_snapshot_schedule" "default" {
  identifier = "terra-redshift-snapshot-schedule"
  recurrence = "cron(0 0/12 * * ? *)" # Every 12 hours
}

# Associate the Snapshot Schedule with the Redshift Cluster
resource "aws_redshift_snapshot_schedule_association" "default" {
  cluster_identifier = aws_redshift_cluster.default.cluster_identifier
  schedule_identifier = aws_redshift_snapshot_schedule.default.identifier
}