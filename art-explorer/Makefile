# Inverse Rendering Pipeline
# Usage: make [target]

P := python
N := 2000
ITERS := 5

# === Pipeline ===
all: generate train validate

iterate:
	@for i in $$(seq 1 $(ITERS)); do \
		echo ""; \
		echo "========== Iteration $$i/$(ITERS) =========="; \
		echo ""; \
		rm -rf data/ models/ validation/; \
		$(P) generate_data.py -n $(N); \
		$(P) train.py --epochs 10; \
		$(P) validate.py; \
	done
	@echo ""
	@echo "Done. refs.jsonl updated with good predictions."

# === 1. Generate ===
generate:
	$(P) generate_data.py -n $(N)

generate-preview:
	$(P) generate_data.py -n $(N) --clean --preview

generate-biased:
	$(P) generate_data.py --scored -n 100

# === 2. Train ===
train:
	$(P) train.py --epochs 10

train-long:
	$(P) train.py --epochs 30

# === 3. Validate ===
validate:
	$(P) validate.py

validate-image:
	$(P) validate.py --image $(IMG)

# === Refs ===
save-ref:
	$(P) save_ref.py add $(ID)

rm-ref:
	$(P) save_ref.py rm $(LINE)

list-refs:
	$(P) save_ref.py list

score-refs:
	$(P) score_refs.py

score-refs-fresh:
	$(P) score_refs.py --rescore

# === Clean ===
clean-data:
	rm -rf data/ data-scored/

clean-models:
	rm -rf models/

clean-all: clean-data clean-models
	rm -rf validation/

# === Help ===
help:
	@echo "Inverse Rendering Pipeline"
	@echo ""
	@echo "Pipeline:   make all              # generate → train → validate"
	@echo "            make iterate           # feedback loop (generate/train/validate × ITERS)"
	@echo "            make iterate ITERS=10 N=1000"
	@echo ""
	@echo "1. Generate: make generate         # random + ref-biased pairs → data/"
	@echo "             make generate N=5000"
	@echo "             make generate-preview  # serial w/ preview window"
	@echo "             make generate-biased   # VLM-weighted breeding → data-scored/"
	@echo ""
	@echo "2. Train:    make train            # 10 epochs"
	@echo "             make train-long       # 30 epochs"
	@echo ""
	@echo "3. Validate: make validate         # predict → render → SSIM, good → refs.jsonl"
	@echo "             make validate-image IMG=path/to/img.png"
	@echo ""
	@echo "Refs:        make save-ref ID=328  # add sample to refs.jsonl"
	@echo "             make save-ref ID=bias:42  # from data-scored/"
	@echo "             make rm-ref LINE=3    # remove by line number"
	@echo "             make list-refs"
	@echo "             make score-refs       # VLM rank refs (Qwen2-VL)"
	@echo "             make score-refs-fresh # re-score all"
	@echo ""
	@echo "Clean:       make clean-data | clean-models | clean-all"

.PHONY: all iterate \
        generate generate-preview generate-biased \
        train train-long \
        validate validate-image \
        save-ref rm-ref list-refs score-refs score-refs-fresh \
        clean-data clean-models clean-all help
