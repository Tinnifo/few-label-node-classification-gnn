FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime

WORKDIR /workdir

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
