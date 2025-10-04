@echo off
echo Stopping and removing containers...
docker compose down

echo Building and starting containers...
docker compose up -d --build

echo Done! Containers are starting up.
echo Run 'docker compose logs -f' to see logs.
