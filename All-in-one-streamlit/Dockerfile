FROM public.ecr.aws/docker/library/python:3.9-slim
WORKDIR /opt/app
RUN apt update && apt upgrade -y && apt-get install -y curl libgl1 libglib2.0-0
RUN pip install --upgrade pip 
COPY requirements.txt /opt/app/
RUN pip install -r requirements.txt
COPY app2.py /opt/app/ 
EXPOSE 8501 
ENTRYPOINT ["streamlit", "run", "app2.py", "--server.port=80"]
