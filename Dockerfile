FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY requirements.txt pyproject.toml ./
RUN uv pip install --system --no-cache -r requirements.txt

COPY . .

# Run train to compile local model artifacts
RUN python -m src.train

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
