.PHONY: setup db stop dev test logs clean reset

setup:
	cp -n .env.example .env || true
	pip install -r requirements.txt

db:
	docker compose up -d

stop:
	docker compose down

dev:
	uvicorn backend.main:app --reload

test:
	pytest

logs:
	tail -f logs/app.log

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache logs/

reset: stop db
