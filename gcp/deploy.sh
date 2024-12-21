export $(grep -v '^#' .env | xargs)
gcloud functions deploy pandas-gcp --entry-point call_python --set-env-vars OPENAI_API_KEY=${OPENAI_API_KEY} --runtime python312 --trigger-http

#gcloud functions deploy pandas-gcp-test --entry-point pandas_http --runtime python312 --trigger-http