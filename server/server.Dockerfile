# Use Python base image
FROM python:3.11-slim

WORKDIR /app/server

COPY server/ .             
COPY ../utils.py /app/utils.py       
COPY ../requirements.txt requirements.txt

# Install dependencies
RUN pip install -r requirements.txt

# Expose server port
EXPOSE 8080

# Run the Flower server
CMD ["python", "server.py"]
