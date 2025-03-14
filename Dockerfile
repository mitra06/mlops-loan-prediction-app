# Use official Python image
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy only requirements file first
COPY requirements.txt /app/


# Upgrade pip first
RUN pip install --upgrade pip

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . /app


# Expose the FastAPI port
EXPOSE 8000

# Run FastAPI with Uvicorn
CMD ["uvicorn", "src.model_predict:app", "--host", "0.0.0.0", "--port", "8000"]
