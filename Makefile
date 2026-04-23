.PHONY: up down logs lint-backend lint-frontend

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f

lint-backend:
	ruff check .
	ruff format --check .

lint-frontend:
	cd frontend && npm run lint