from dotenv import load_dotenv
import os

# Load .env variables
if os.path.isfile('.env'):
    load_dotenv('.env')

DIFFBOT_TOKEN = os.environ.get('DIFFBOT_TOKEN')
