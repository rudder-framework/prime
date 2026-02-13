FROM python:3.11-slim

WORKDIR /app

COPY framework/__init__.py ./framework/__init__.py
COPY framework/explorer/ ./framework/explorer/
COPY framework/sql/ ./framework/sql/
COPY data/ ./data/

EXPOSE 8080

CMD ["python", "-m", "framework.explorer.server", "--port", "8080", "data"]
