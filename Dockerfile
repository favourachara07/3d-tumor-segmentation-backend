# Start from an official Python image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /code

# Copy the requirements file and install dependencies
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy your application code into the container
COPY ./app /code/app
COPY ./models /code/models # Copy the placeholder

# Expose the port the app will run on
EXPOSE 8000

# Command to run the Uvicorn server
# The --host 0.0.0.0 is crucial to make it accessible within the Space
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]