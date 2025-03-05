resource "random_id" "suffix" {
  byte_length = 4
}

resource "random_password" "db" {
  length  = 16
  special = false
}

resource "aws_db_instance" "default" {
  allocated_storage   = 20
  engine              = "mysql"
  engine_version      = "8.0"
  instance_class      = "db.t3.micro"
  name                = "mydb"
  username            = "admin"
  password            = random_password.db.result
  skip_final_snapshot = true
  storage_type        = "gp3"
  identifier          = "mydb-${random_id.suffix.hex}"
}