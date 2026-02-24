FROM python:3.11-slim

WORKDIR /app

COPY index.html /app/index.html
COPY styles.css /app/styles.css
COPY app.js /app/app.js
COPY app.js.map /app/app.js.map
COPY input_contract.json /app/input_contract.json

EXPOSE 8080
CMD ["sh", "-c", "python3 -m http.server ${PORT:-8080} --directory /app"]
