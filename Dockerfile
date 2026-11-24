FROM python:3.12-slim

COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.8.4 /lambda-adapter /opt/extensions/lambda-adapter

RUN pip install --no-cache-dir uv==0.8.13

WORKDIR /code

COPY ./pyproject.toml ./README.md ./uv.lock* ./
COPY ./app ./app

RUN uv pip install --system -r pyproject.toml

ENV PORT=8080
EXPOSE 8080

CMD ["uvicorn", "app.web_server:app", "--host", "0.0.0.0", "--port", "8080"]