# APG — Autonomous Parallel Code Generation
# Makefile for development workflow

.PHONY: help check run worker test submit clean

help:
	@echo "APG Development Commands"
	@echo "========================"
	@echo "  make check          Check all system dependencies"
	@echo "  make run            Start the Go orchestrator (HTTP API)"
	@echo "  make worker         Start the Python NATS worker"
	@echo "  make test           Run all tests (Go + Python)"
	@echo "  make submit FILE=   Submit a C file to the pipeline"
	@echo "  make clean          Remove build artifacts"
	@echo ""
	@echo "Quick start:"
	@echo "  1. docker compose up -d        (start NATS)"
	@echo "  2. make worker                 (Python worker, separate terminal)"
	@echo "  3. make run                    (Go orchestrator, separate terminal)"
	@echo "  4. make submit FILE=examples/dot_product.c"

check:
	python scripts/check_env.py

run:
	go run ./cmd/orchestrator \
		--addr :8080 \
		--db data/apg.db \
		--nats nats://localhost:4222

worker:
	python -m workers.nats_worker

# Run pipeline directly without NATS (good for testing)
pipeline:
	@test -n "$(FILE)" || (echo "Usage: make pipeline FILE=examples/vector_scale.c" && exit 1)
	python scripts/run_pipeline.py \
		--source $(FILE) \
		--provider ollama \
		--max-candidates 3 \
		--max-rounds 3

test:
	go test ./... -v
	python -m pytest tests/python/ -v

submit:
	@test -n "$(FILE)" || (echo "Usage: make submit FILE=examples/vector_scale.c" && exit 1)
	curl -s -X POST http://localhost:8080/submit \
		--data-binary "@$(FILE)" | python -m json.tool

result:
	@test -n "$(ID)" || (echo "Usage: make result ID=<job-id>" && exit 1)
	curl -s http://localhost:8080/result/$(ID) | python -m json.tool

treesitter-test:
	python workers/code_understanding/treesitter_t1.py

verifier-test:
	python workers/spec_generator/verifier.py

clean:
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -f data/apg.db
