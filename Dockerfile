# Step 1: Use an official Python runtime as a parent image
FROM python:3.9-slim

# Step 2: Set the working directory in the container
WORKDIR /app

# Step 3: Copy the current directory contents into the container at /app
COPY . /app

# Step 4: Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Expose the port Streamlit runs on (default is 8501)
EXPOSE 8501

# Step 6: Define environment variables
COPY .env /app/.env

# Step 7: Run Streamlit app
CMD ["streamlit", "run", "main.py"]
