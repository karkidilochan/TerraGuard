# Use an official Python runtime as the base image
FROM python:3.11-slim

# Set working directory in the container
WORKDIR /app

# Install Poetry
RUN pip install poetry

# Copy only the dependency files to leverage Docker caching
COPY pyproject.toml poetry.lock* /app/

# Install dependencies (excluding dev dependencies for production)
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

# Copy the rest of the project files
COPY app/ ./app

# Expose port (e.g., for FastAPI with Uvicorn)
EXPOSE 8000

# Command to run the application
# CMD ["python", "-m", "app.rag_pipeline"]
# Start an interactive shell
CMD ["/bin/bash"]

# docker run -it -v $(pwd):/app terraguard   