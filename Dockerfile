FROM python:3.11-slim

WORKDIR /app

# Copy static files
COPY orthon/static/ ./static/

# Railway injects PORT env var
ENV PORT=8000

# Serve static HTML
CMD python -m http.server $PORT --directory static
