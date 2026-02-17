# Art Explorer Makefile (minimal)

P := python
N ?= 5000
Q ?= 1000

# Core workflow
generate:
	$(P) generate.py -n $(N)

bootstrap:
	$(P) score_refs.py
	$(P) generate.py --scored -n $(N) --out data/quality

train-super:
	$(P) train.py --forward-only --data data
	$(P) train.py --inverse-only --data data data/quality
	$(P) train.py --taste-only --data data data/quality

discover:
	$(P) predict.py --explore --mode balanced

discover-ref:
	$(P) predict.py --explore --mode ref

discover-novel:
	$(P) predict.py --explore --mode novel

loop:
	$(P) score_refs.py
	$(P) generate.py --scored -n $(Q) --out data/quality
	$(P) train.py --inverse-only --data data data/quality
	$(P) train.py --taste-only --data data data/quality
	$(P) predict.py --explore --mode balanced -n 20 --candidates 500

# Ref curation
save-ref:
	$(P) save_ref.py add $(ID)

save-ref-quality:
	$(P) save_ref.py add q:$(ID)

rm-ref:
	$(P) save_ref.py rm $(LINE)

list-refs:
	$(P) save_ref.py list

# Utilities
clean:
	rm -rf data/ output/ models/

help:
	@echo "Art Explorer"
	@echo ""
	@echo "  make generate N=5000   # coverage data (param mastery)"
	@echo "  make bootstrap          # refs -> scored refs + quality data"
	@echo "  make train-super        # train forward + inverse + taste"
	@echo "  make discover           # balanced explore"
	@echo "  make loop Q=1000        # refinement round"
	@echo ""
	@echo "Curation:"
	@echo "  make save-ref ID=out:0"
	@echo "  make save-ref-quality ID=123"
	@echo "  make list-refs"
	@echo "  make rm-ref LINE=3"

.PHONY: generate bootstrap train-super discover discover-ref discover-novel loop \
        save-ref save-ref-quality rm-ref list-refs clean help
