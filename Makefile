# bb-llm

P := python
ART := art-explorer
TRAIN := training-tests

# Art Explorer
art:         ; cd $(ART) && $(P) explore.py --gens 50 --pop 8
art-preview: ; cd $(ART) && $(P) explore.py --gens 50 --pop 8 --preview
art-score:   ; cd $(ART) && $(P) score.py $(IMG)
art-cleanup:
	@-pkill -9 -f "python.*explore" 2>/dev/null || true
	@rm -f $(ART)/art_data/ratings.jsonl
	@rm -rf $(ART)/art_data/screenshots/*
	@echo "Cleaned up"

# Training
train: ; cd $(TRAIN) && $(P) train_gpt.py
infer: ; cd $(TRAIN) && $(P) infer.py
demo:  ; cd $(TRAIN) && $(P) demo.py
