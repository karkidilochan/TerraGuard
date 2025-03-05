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
  instance_class      = "db.t2.micro"
  name                = "mydb"
  username            = "admin"
  password            = random_password.db.result
  db_subnet_group_name   = aws_db_subnet_group.default.name
  vpc_security_group_ids = [aws_security_group.default.id]
  storage_type        = "gp3"
  identifier          = "mydb-${random_id.suffix.hex}"
  skip_final_snapshot = true
}

resource "aws_db_subnet_group" "default" {
  name       = "main"
  subnet_ids = ["subnet-0bb1c79de3EXAMPLE", "subnet-069b4f69EXAMPLE"]
}

resource "aws_security_group" "default" {
  name        = "allow_tls"
  description = "Allow TLS inbound traffic"

  ingress {
    description = "TLS from VPC"
    from_port   = 3306
    to_port     = 3306
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/16"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}