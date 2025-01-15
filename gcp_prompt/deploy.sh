# --service-account=autograde-submission@autograde-314802.iam.gserviceaccount.com
export $(grep -v '^#' .env | xargs)
gcloud functions deploy validate_code --runtime python312 --trigger-http --timeout=120 --allow-unauthenticated --set-env-vars OPENAI_GPT_MODEL=gpt-4o-mini --set-env-vars TEST_CASES_URL="https://raw.githubusercontent.com/alexlopespereira/ipynb-autograde/refs/heads/master/data/questions.json"


