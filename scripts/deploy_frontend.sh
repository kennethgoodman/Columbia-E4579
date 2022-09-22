#!/bin/bash
cd services/frontend
export REACT_APP_API_URL=/api
npm run i
npm run build
aws s3 sync build/ s3://columbia-e4579-frontend
