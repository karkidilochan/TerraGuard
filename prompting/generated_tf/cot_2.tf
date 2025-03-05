resource "aws_redshift_cluster" "default" {
  cluster_identifier      = "redshift-cluster-1"
  database_name           = "mydb"
  master_username         = "foo"
  master_password         = "Mustbe14chars"
  node_type               = "dc2.large"
  cluster_type            = "single-node"
  automated_snapshot_start_time = "12:00"
}