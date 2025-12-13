## sxmi-test utilities

### Environment
- Use the existing [YARP environment](https://github.com/Savoie-Research-Group/yarp)

### What’s here
- `bnfm_iterator.py` / `bnfm_iterator_batch.py`: iterative BNFM enumeration (b0f0 ~ b3f3 by default), with local dedup per reactant and optional parallel execution.
- `postprocess_bnfm.py`: deduplicate reactions/products from the iterator output and generate reaction SMILES.
- `run_single_smiles.sh`: convenience wrapper to pick a SMILES line from `unique_smiles.txt` and run the iterator.

### Typical usage
- Single run:
  ```bash
  python bnfm_iterator_batch.py --smiles "C=CC=C" --max-break 3 --max-form 3 --output bnfm_results.pkl --log-file bnfm.log --max-workers 4
  ```
  - `--max-break` / `--max-form`: enumerate all b0..bN and f0..fM combos.
  - `--score-threshold`: filter on `bond_mat_scores[0]` (default 0.0).
  - `--max-workers`: parallel reactant-level enumeration.
  - `--output` / `--log-file`: paths for pkl/log.

- Batch via `run_single_smiles.sh` (uses `unique_smiles.txt`):
  ```bash
  ./run_single_smiles.sh 42
  ```

- Post-process:
  ```bash
  python postprocess_bnfm.py --input bnfm_results.pkl --output bnfm_cleaned.pkl
  ```

### Notes
- [autoCG](https://github.com/Romarin87/autoCG) can be used for generating 3D.

## Modifications relative to upstream YARP
- Scoring weights: `w_aro` adjusted from `-24` to `-2.4`.
- Carbene bonus: recognize carbon with element `c`, formal charge `0`, and 6 valence electrons as a carbene; add a bonus of `-4.931` to the score when present.

## BNFM filtering additions
Within `bnfm_iterator_batch.py` products are filtered with:
- Drop if any bond order > 3.
- Drop if any formal charge has |charge| > 1.
- Drop if more than two atoms carry non-zero formal charges.
- Drop if two charged atoms are not directly bonded to each other.

## Recommended convergence threshold
Use a score threshold of `0.15` to retain carbene (~0.149), CO(~0.131), and isocyanate(~0.077) non-standard Lewis structures, while block N2O(~0.203), N1C=C1(~0.800).
