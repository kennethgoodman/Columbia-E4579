#!/bin/sh

gunicorn -b 0.0.0.0:5000 wsgi:app --access-logfile - --workers 1 --threads 5 --timeout 120 --log-level=debug
