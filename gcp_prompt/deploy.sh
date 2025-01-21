# --service-account=autograde-submission@autograde-314802.iam.gserviceaccount.com
#export $(grep -v '^#' .env | xargs)
gcloud functions deploy validate_code --runtime python312 --memory=512MB --trigger-http --timeout=120 --allow-unauthenticated --set-env-vars OPENAI_GPT_MODEL=gpt-4o-mini --set-env-vars EMAIL_HASH_SALT="aambaenap15" --set-env-vars TEST_CASES_URL="https://raw.githubusercontent.com/alexlopespereira/ipynb-autograde/refs/heads/master/data/questions.json"


