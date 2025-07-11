import os

from dotenv import load_dotenv
from vertexai import agent_engines, init

from recap_agent import root_agent

load_dotenv()
PROJECT_ID = os.environ["GOOGLE_CLOUD_PROJECT"]
LOCATION = os.environ["GOOGLE_CLOUD_LOCATION"]
STAGING_BUCKET = os.environ["GOOGLE_CLOUD_STAGING_BUCKET"]
print(PROJECT_ID)
print(LOCATION)
print(STAGING_BUCKET)
init(
    project=PROJECT_ID,
    location=LOCATION,
    staging_bucket=STAGING_BUCKET,
)

remote_app = agent_engines.create(
    agent_engine=root_agent,
    requirements=[
        "google-adk (>=0.0.2)",
        "google-cloud-aiplatform[agent_engines] (>=1.91.0,!=1.92.0)",
        "google-genai (>=1.5.0,<2.0.0)",
        "dotenv (>=0.9.9)"
    ]
)

