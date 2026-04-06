P := python
N ?= 5000
THRESHOLD ?= -0.3

# ── Data ─────────────────────────────────────────────────

generate:
	$(P) generate.py -n $(N)

DATA ?= data

caption:
	$(P) caption.py --data $(DATA) --workers 4

train:
	$(P) train.py --data data

eval:
	$(P) eval_sft.py --data data -n 50

eval-render:
	$(P) eval_sft.py --data data --render -n 50

# ── Inference ─────────────────────────────────────────────

predict:
	$(P) infer.py --target $(TARGET)

predict-refine:
	$(P) infer.py --target $(TARGET) --refine

text:
	$(P) infer.py --text "$(TEXT)"

# ── Curation ──────────────────────────────────────────────

judge:
	$(P) judge.py -n 10 --threshold 80

save-ref:
	$(P) save_ref.py add $(ID)

rm-ref:
	$(P) save_ref.py rm $(LINE)

list-refs:
	$(P) save_ref.py list

thumb-refs:
	$(P) thumb_refs.py

capture:
	$(P) capture.py $(ID)

# ── RL ────────────────────────────────────────────────────

rl:
	$(P) rl.py

# rl (auto-harvests) -> caption -> train -> eval
round:
	$(MAKE) rl
	$(MAKE) caption
	$(MAKE) train
	$(MAKE) eval

# ── Dashboard ─────────────────────────────────────

dash:
	$(P) -m uvicorn serve:app --reload --port 8000

# ── Misc ──────────────────────────────────────────────────

clean:
	rm -rf data/images data/params output/ models/

help:
	@echo "  make caption               caption uncaptioned images"
	@echo "  make train                 SFT on data/"
	@echo "  make round                 full cycle: rl -> caption -> train -> eval"
	@echo "  make rl                    GRPO (add JUDGE=1 for aesthetic scoring)"
	@echo "  make eval                  metrics + coverage"
	@echo ""
	@echo "  make predict TARGET=x.png  image -> params -> render"
	@echo "  make text TEXT='spiral...' text -> params -> render"
	@echo "  make judge                 VLM discovery + scoring"
	@echo ""
	@echo "  make save-ref ID=328       add to refs"
	@echo "  make rm-ref LINE=3         remove ref"
	@echo "  make list-refs             list refs"
	@echo "  make capture ID=328        hi-res capture"
	@echo ""
	@echo "  make generate N=5000       new base data (if needed)"
	@echo ""
	@echo "  make dash                  dashboard API (port 8000; run: pip install -r requirements.txt)"

.PHONY: generate caption train eval eval-render \
        predict predict-refine text \
        judge save-ref rm-ref list-refs thumb-refs capture \
        rl round clean help dash
