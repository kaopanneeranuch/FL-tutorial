# Use Python base image
FROM python:3.11-slim

WORKDIR /app/client

COPY ../requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY ../utils.py /app/utils.py

COPY . .
COPY ../dataset/part0.pt ./client_data/part0.pt
COPY ../dataset/part1.pt ./client_data/part1.pt
COPY ../dataset/part2.pt ./client_data/part2.pt
COPY ../dataset/testset.pt ./client_data/testset.pt

# Set environment variables
ENV PARTITION_ID=0

# Set the default command to run the client script
CMD ["python", "client.py"]
