# Art Explorer Makefile

P := python
N ?= 5000

# Core workflow
generate:
	$(P) generate.py -n $(N)

train:
	$(P) train.py --data data

discover:
	$(P) predict.py --explore --mode balanced

discover-ref:
	$(P) predict.py --explore --mode ref

discover-novel:
	$(P) predict.py --explore --mode novel

loop:
	$(P) generate.py -n $(N)
	$(P) train.py --data data
	$(P) predict.py --explore --mode balanced -n 20 --candidates 500

# Ref curation
save-ref:
	$(P) save_ref.py add $(ID)

open-ref:
	$(P) save_ref.py open $(ID)

rm-ref:
	$(P) save_ref.py rm $(LINE)

list-refs:
	$(P) save_ref.py list

capture:
	$(P) capture.py --line $(LINE)

thumb-refs:
	$(P) thumb_refs.py

# Utilities
clean:
	rm -rf data/ output/ models/

help:
	@echo "Art Explorer"
	@echo ""
	@echo "  make generate N=5000   # coverage data"
	@echo "  make train             # forward + inverse + taste"
	@echo "  make discover          # balanced explore"
	@echo "  make loop N=5000       # generate + train + discover"
	@echo ""
	@echo "Curation:"
	@echo "  make save-ref ID=328"
	@echo "  make save-ref ID=out:0"
	@echo "  make open-ref ID=328"
	@echo "  make list-refs"
	@echo "  make rm-ref LINE=3"
	@echo "  make thumb-refs        # cache refs thumbs to data/refs/images"

.PHONY: generate train discover discover-ref discover-novel loop \
        save-ref open-ref rm-ref list-refs capture thumb-refs clean help
