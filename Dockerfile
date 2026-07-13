services:
  api:
    build: .
    ports:
      - "8000:8000"
    env_file:
      - .env
    depends_on:
      - redis
    volumes:
      - ./data:/app/data

  dashboard:
    build: .
    command: streamlit run app/dashboard.py --server.port 8501 --server.address 0.0.0.0
    ports:
      - "8501:8501"
    env_file:
      - .env
    depends_on:
      - api

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    command: redis-server --maxmemory 256mb --maxmemory-policy allkeys-lru