# Use Python base image
FROM python:3.11-slim

WORKDIR /app/server

COPY ../requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY ../utils.py /app/utils.py 

COPY . .      

# Expose server port
EXPOSE 8080

# Run the Flower server
CMD ["python", "server.py"]
