FROM python:3.6

# install build utilities
RUN apt-get update && \
	apt-get install -y gcc make apt-transport-https ca-certificates build-essential

# check our python environment
RUN python3 --version
RUN pip3 --version

# set the working directory for containers
WORKDIR  /usr/src/poker-bot

# Installing python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all the files to the working directory
COPY game.py .
COPY /agent/* .
COPY /own_models/ ./own_models
RUN ls -la /

# Running Python Application
#CMD ["python3", "game.py","-env","leduc"]
#CMD ["python3", "game.py","-env","limit"]
ENTRYPOINT ["python", "game.py"]