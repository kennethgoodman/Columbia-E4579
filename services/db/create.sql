UPDATE mysql.user SET Grant_priv='Y', Super_priv='Y' WHERE User='root';
FLUSH PRIVILEGES;
GRANT ALL ON *.* TO 'mysql'@'%';
CREATE DATABASE api_dev;
CREATE DATABASE api_test;
