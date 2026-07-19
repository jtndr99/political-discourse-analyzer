import os
import sys

# Ensure the parent directory is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mangum import Mangum
from app.web_server import app

# This is the entry point for AWS Lambda
handler = Mangum(app, lifespan="off")
