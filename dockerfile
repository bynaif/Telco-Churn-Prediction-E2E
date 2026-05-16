FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY model/ model/
COPY src/ src/
COPY start.sh .
CMD ["bash", "start.sh"]