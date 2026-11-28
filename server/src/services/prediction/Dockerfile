# Install Python 
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY server/src/services/prediction/requirements.txt ./requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy prediction service
COPY server/src/services/prediction/prediction_service.py ./prediction_service.py

# Copy PredictionModel submodule
COPY PredictionModel ./PredictionModel

# Flush print() output immediately so startup messages appear in container logs
ENV PYTHONUNBUFFERED=1

# Model location inside container
ENV MODEL_PATH=/app/PredictionModel/src/Bayesian_Model/Bayesian_Prediction_Model.pkl

EXPOSE 5001

# Start the application on $PORT (hosts like Railway inject it), defaulting to 5001
CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT:-5001} --timeout 120 prediction_service:app"]