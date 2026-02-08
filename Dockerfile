FROM python:3.11-slim

WORKDIR /app

COPY orthon/__init__.py ./orthon/__init__.py
COPY orthon/explorer/ ./orthon/explorer/
COPY orthon/sql/ ./orthon/sql/
COPY data/ ./data/

EXPOSE 8080

CMD ["python", "-m", "orthon.explorer.server", "--port", "8080", "data"]
