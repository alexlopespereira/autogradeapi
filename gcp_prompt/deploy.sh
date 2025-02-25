# --service-account=autograde-submission@autograde-314802.iam.gserviceaccount.com
#export $(grep -v '^#' .env | xargs)
gcloud functions deploy validate_code --runtime python312 --memory=512MB --trigger-http --timeout=240 --allow-unauthenticated --set-env-vars OPENAI_GPT_MODEL=o3-mini-2025-01-31 --set-env-vars LLM_PROVIDER=OPENAI --set-env-vars TEST_CASES_URL="https://raw.githubusercontent.com/alexlopespereira/ipynb-autograde/refs/heads/master/data/questions.json"


