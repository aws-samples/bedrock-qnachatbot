# Use the official Python image as base
FROM python:3.10.13

ENV HOST=0.0.0.0
 
ENV LISTEN_PORT 8080
 
EXPOSE 8080

# Set the AWS_DEFAULT_REGION environment variable
ENV AWS_DEFAULT_REGION=us-east-1

# Define build arguments for access key and secret access key
ARG AWS_REGION_NAME
ARG S3_BUCKET_NAME

# Set the AWS credentials environment variables
#ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
#ENV AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
ENV AWS_DEFAULT_REGION=$AWS_REGION_NAME
ENV S3_BUCKET_NAME=$S3_BUCKET_NAME

# Set the working directory inside the container
WORKDIR /app

    
# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install all dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY ./Assets /app/Assets
COPY ./src /app/src
COPY loki-config.ini .
#COPY ./vectorstore /app/vectorstore
COPY run.sh .
RUN chmod +x /app/run.sh

# Run bash Script
RUN /bin/sh run.sh $S3_BUCKET_NAME $AWS_REGION_NAME

CMD ["/bin/sh", "-c", "export"]
# Command to run the application
ENTRYPOINT ["streamlit", "run"]
CMD ["src/1_🏠_Home.py","--server.port", "8080"]