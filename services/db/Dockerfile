# pull official base image
FROM mysql:8

# run create.sql on init
ADD 00_create.sql /docker-entrypoint-initdb.d/00_create.sql
ADD 01_user.sql /docker-entrypoint-initdb.d/01_user.sql
ADD 02_rest.sql /docker-entrypoint-initdb.d/02_rest.sql
