set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

default:
  @just --list

generate N="5000":
  cd "{{justfile_directory()}}" && python generate.py -n {{N}}

caption DATA="data":
  cd "{{justfile_directory()}}" && python caption.py --data {{DATA}} --workers 4

train:
  cd "{{justfile_directory()}}" && python train.py --data data

eval:
  cd "{{justfile_directory()}}" && python eval_sft.py --data data -n 50

eval-render:
  cd "{{justfile_directory()}}" && python eval_sft.py --data data --render -n 50

predict TARGET:
  cd "{{justfile_directory()}}" && python infer.py --target {{TARGET}}

predict-refine TARGET:
  cd "{{justfile_directory()}}" && python infer.py --target {{TARGET}} --refine

text TEXT:
  cd "{{justfile_directory()}}" && python infer.py --text "{{TEXT}}"

judge:
  cd "{{justfile_directory()}}" && python judge.py -n 10 --threshold 80

discover:
  just judge

save-ref ID:
  cd "{{justfile_directory()}}" && python save_ref.py add {{ID}}

rm-ref LINE:
  cd "{{justfile_directory()}}" && python save_ref.py rm {{LINE}}

list-refs:
  cd "{{justfile_directory()}}" && python save_ref.py list

thumb-refs:
  cd "{{justfile_directory()}}" && python thumb_refs.py

capture ID:
  cd "{{justfile_directory()}}" && python capture.py {{ID}}

rl:
  cd "{{justfile_directory()}}" && python rl.py

bootstrap N="5000" DATA="data":
  just generate N={{N}}
  just caption DATA={{DATA}}
  just train
  just eval

round DATA="data":
  just rl
  just caption DATA={{DATA}}
  just train
  just eval

clean:
  rm -rf "{{justfile_directory()}}/data/images" "{{justfile_directory()}}/data/params" "{{justfile_directory()}}/output" "{{justfile_directory()}}/models"

help:
  @just --list
