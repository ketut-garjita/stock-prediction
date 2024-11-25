FROM python:3.11.9

# Set working directory
WORKDIR /app

# Copy project files to container
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose port 9696
EXPOSE 9696

# Run the Flask app
CMD ["python", "app.py"]

