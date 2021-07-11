FROM ubuntu
ENV TZ=US/Eastern
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && \
    echo $TZ > /etc/timezone
RUN apt-get update
RUN apt-get install -y python3.8 python3-pip python3-dev
RUN python3 -m pip install --upgrade pip==20.1.1
RUN python3 -m pip install pandas==1.0.5 \
    dash==1.13.3 \
    gunicorn==20.0.4
COPY requirements.txt /tmp/
RUN pip install --requirement /tmp/requirements.txt
COPY . /tmp/
ADD . /
EXPOSE 8080
CMD ["python3", "app.py"]