# Inverse Rendering Pipeline
# Usage: make [target]

P := python
N := 2000
ITERS := 5

# === Full Lifecycle ===
all: generate train validate

# === Iterative Training (feedback loop) ===
iterate:
	@for i in $$(seq 1 $(ITERS)); do \
		echo ""; \
		echo "========== Iteration $$i/$(ITERS) =========="; \
		echo ""; \
		rm -rf data/ models/ validation/; \
		$(P) generate_data.py -n $(N); \
		$(P) train_inverse.py --epochs 10; \
		$(P) validate.py; \
	done
	@echo ""
	@echo "Done. refs.jsonl updated with good predictions."

# === Phase 1: Data Generation ===
generate:
	$(P) generate_data.py -n $(N)

generate-preview:
	$(P) generate_data.py -n $(N) --clean --preview

generate-more:
	$(P) generate_data.py -n $(N)

# === Phase 2: Training ===
train:
	$(P) train_inverse.py --epochs 10

train-long:
	$(P) train_inverse.py --epochs 30

# === Phase 3: Validation ===
validate:
	$(P) validate.py

validate-image:
	$(P) validate.py --image $(IMG)

# === Legacy (CMA-ES + VLM) ===
explore:
	$(P) explore.py --gens 50 --pop 8

explore-preview:
	$(P) explore.py --gens 50 --pop 8 --preview

score:
	$(P) score.py $(IMG)

# === Curation ===
save-ref:
	$(P) save_ref.py add $(ID)

rm-ref:
	$(P) save_ref.py rm $(LINE)

list-refs:
	$(P) save_ref.py list

# === Cleanup ===
clean-data:
	rm -rf data/

clean-models:
	rm -rf models/

clean-validation:
	rm -rf validation/

clean-all: clean-data clean-models clean-validation
	rm -rf art_data/

# === Info ===
help:
	@echo "Inverse Rendering Pipeline"
	@echo ""
	@echo "Lifecycle:  make all          # generate → train → validate"
	@echo "            make iterate      # feedback loop (5 iters, cleans each round)"
	@echo "            make iterate ITERS=10 N=1000"
	@echo ""
	@echo "Phase 1:    make generate     # 2k samples"
	@echo "            make generate N=5000"
	@echo "            make generate-preview"
	@echo ""
	@echo "Phase 2:    make train        # 10 epochs"
	@echo "            make train-long   # 30 epochs"
	@echo ""
	@echo "Phase 3:    make validate     # writes good params to refs.jsonl"
	@echo "            make validate-image IMG=path/to/img.png"
	@echo ""
	@echo "Curation:   make save-ref ID=328     # add sample to refs.jsonl"
	@echo "            make rm-ref LINE=3       # remove by line number"
	@echo "            make list-refs           # show all refs"
	@echo ""
	@echo "Legacy:     make explore      # CMA-ES + VLM"
	@echo "            make score IMG=path/to/img.png"
	@echo ""
	@echo "Cleanup:    make clean-data | clean-models | clean-all"

.PHONY: all generate generate-preview generate-more train train-long \
        validate validate-image explore explore-preview score iterate \
        save-ref rm-ref list-refs clean-data clean-models clean-validation clean-all help
