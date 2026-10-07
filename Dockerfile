FROM python:3.9-slim

ARG MODEL_PATH
ARG LABELS_PATH 

ENV MODEL_PATH=${MODEL_PATH}

RUN echo 'MODEL PATH IS' ${MODEL_PATH}
RUN groupadd -g 1000 user && useradd -u 1000 -g 1000 -m user

RUN chown -R user:user /home/user
WORKDIR /home/user

COPY models /home/user/models
COPY util /home/user/util
COPY datasets /home/user/
COPY runner /home/user/runner
COPY requirements.txt /home/user/
COPY main.py /home/user/
COPY config.yaml /home/user/

COPY ${MODEL_PATH} /home/user/${MODEL_PATH}
COPY ${LABELS_PATH} /home/user/${LABELS_PATH}
RUN pip install --ignore-installed -r requirements.txt --no-cache-dir

USER 1000

ENTRYPOINT ["gunicorn", "-b", ":8000", "runner.wsgi:app"]

EXPOSE 8000