# Use Python 3.10 slim base image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy dependencies list
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files into container
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Run API with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
