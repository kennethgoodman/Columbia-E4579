#!/bin/bash
cd services/client
export REACT_APP_API_SERVICE_URL=/api
npm install
npm install react-scripts
npm run build
aws s3 sync build/ s3://columbia-e4579-frontend
