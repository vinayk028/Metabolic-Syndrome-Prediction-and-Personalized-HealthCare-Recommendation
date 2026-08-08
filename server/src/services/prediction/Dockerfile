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

# Model location inside container
ENV MODEL_PATH=/app/PredictionModel/src/Bayesian_Model/Bayesian_Prediction_Model.pkl

EXPOSE 5001

# Start the application
# CMD ["python", "prediction_service.py"]
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "prediction_service:app"]