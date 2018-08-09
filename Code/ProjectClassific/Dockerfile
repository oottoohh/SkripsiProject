FROM python:3

# set working directory

ADD . /ProjectClassific

WORKDIR /ProjectClassific

RUN 	pip install --upgrade pip==18.0 && \
	pip install --no-cache-dir -r requirements.txt

EXPOSE 8081

COPY . /ProjectClassific/app

CMD [ "python", "./app.py" ]