export $(grep -v '^#' .env | xargs)
gcloud functions deploy pandas-gcp --entry-point call_python --runtime python312 --trigger-http --timeout=120

#gcloud functions deploy pandas-gcp-test --entry-point pandas_http --runtime python312 --trigger-http