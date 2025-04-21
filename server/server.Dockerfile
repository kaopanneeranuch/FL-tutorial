# Use Python base image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app/server

# Copy only server folder content
COPY server/ .             
COPY ../utils.py ../       

# Copy shared requirements if any
COPY ../requirements.txt requirements.txt

# Install dependencies
RUN pip install -r requirements.txt

# Expose server port
EXPOSE 8080

# Run the Flower server
CMD ["python", "server.py"]
