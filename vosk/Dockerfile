FROM python:3.9

ADD requirements.txt ./requirements.txt

RUN apt-get update
RUN pip install --upgrade pip && pip install -r requirements.txt

# ADD https://alphacephei.com/vosk/models/vosk-model-ru-0.10.zip .
# RUN unzip vosk-model-ru-0.10.zip && mv vosk-model-ru-0.10 vosk_model
ADD ./model /vosk_model


WORKDIR /app
