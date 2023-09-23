#!/bin/bash

file="services/db/02_rest.sql"

# Checking file size
file_size=$(wc -c < "$file")
expected_size=1194415983
size_tolerance=1000000

if (( file_size < expected_size - size_tolerance || file_size > expected_size + size_tolerance )); then
  echo "Error: File size of $file is not within tolerance of $expected_size bytes"
  exit 1
fi

# Checking line count
line_count=$(wc -l < "$file")
expected_line_count=1274

if (( line_count != expected_line_count )); then
  echo "Error: Line count of $file is not $expected_line_count"
  exit 1
fi

handle_sigint() {
  echo "Received SIGINT, shutting down Docker containers..."
  docker-compose down
  exit 1
}
trap handle_sigint SIGINT

docker-compose -f docker-compose.full_db.yaml up --build --force-recreate 