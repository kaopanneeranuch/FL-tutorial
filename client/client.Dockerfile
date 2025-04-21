FROM python:3.11-slim

WORKDIR /app

COPY client/ .           
COPY dataset/part0.pt ./client_data/part0.pt
COPY dataset/part1.pt ./client_data/part1.pt
COPY dataset/part2.pt ./client_data/part2.pt
COPY dataset/testset.pt ./client_data/testset.pt
COPY ../utils.py ../
COPY ../requirements.txt requirements.txt

RUN pip install -r requirements.txt

ENV PARTITION_ID=0

CMD ["python", "client.py"]