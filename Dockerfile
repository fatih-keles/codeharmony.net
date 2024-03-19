# Use an official Python runtime as a parent image
#FROM python:3.10
FROM python:3.10-slim

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

RUN apt-get update \
    && apt-get install -y wget \
    && apt-get install -y curl \
    && apt-get -y install chromium \ 
    && apt-get clean

# CHROMIUM default flags for container environnement
# The --no-sandbox flag is needed by default since we execute chromium in a root environnement
RUN echo 'export CHROMIUM_FLAGS="$CHROMIUM_FLAGS --no-sandbox"' >> /etc/chromium.d/default-flags

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

ENV FLASK_APP=run.py

# Set the default command to run when the container starts
CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]
# CMD [ "sh"]