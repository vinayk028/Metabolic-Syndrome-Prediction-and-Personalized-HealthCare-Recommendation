# Install Python 
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port the app runs on
EXPOSE 5001

# Start the application
# CMD ["python", "prediction_service.py"]
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "prediction_service:app"]