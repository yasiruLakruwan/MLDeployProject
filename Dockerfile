FROM python:3.8-slim-buster
WORKDIR /app
COPY . /app

RUN sed -i 's|http://deb.debian.org|http://archive.debian.org|g' /etc/apt/sources.list \
    && apt update -y \
    && apt install -y awscli
RUN pip install -r requirments.txt 

CMD [ "python3","app.py" ]
