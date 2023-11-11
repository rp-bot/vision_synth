# Use an official Python runtime as a parent image
FROM python:3.12

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . /usr/src/app

# Install any needed packages specified in requirements.txt
# Uncomment the next line if you have requirements
# RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container
# Uncomment the next line if your app needs a port
# EXPOSE 80

# Define environment variable
# Uncomment the next line if you need environment variables
# ENV NAME World

# Run app.py when the container launches
# Replace 'app.py' with the script you want to run
CMD ["python", "./app.py"]
