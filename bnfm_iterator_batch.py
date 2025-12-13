"""
Iterative BNFM enumeration utilities.

Given one or more SMILES reactants, this module will:
1. Enumerate break-n/form-m products for all n,m in [0, max_break]x[0, max_form].
2. Filter products by Lewis score (bond_mat_scores[0] <= score_threshold).
3. Deduplicate by yarpecule hash locally per reactant (A->B and B->A are both retained),
   serialize results to a pickle.
4. Iterate by using newly generated products as the next round of reactants until no
   additional globally new species are produced.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import pickle
import time
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import yarp as yp
from yarp.yarpecule import yarpecule

MAX_BOND_ORDER = 3
SCRIPT_START = time.time()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def configure_logging(log_file: str) -> None:
    """Configure file and console logging once."""
    if logger.handlers:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    fh = logging.FileHandler(log_file, mode="w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

def yarpecule_to_record(mol: yarpecule) -> Dict[str, object]:
    """Convert a yarpecule into a serializable dictionary."""
    return {
        "adj_mat": mol.adj_mat,
        "elements": mol.elements,
        "q": mol.q,
        # store only the lowest-score structure, but keep list shape to match downstream indexing
        "bond_mats": mol.bond_mats[0],
        "bond_mat_scores": mol.bond_mat_scores[0],
        "hash": mol.hash,
    }


def _form_steps(
    starting: Sequence[yarpecule], steps: int, skip_hashes: Optional[Set[float]]
) -> List[yarpecule]:
    """
    Perform ``steps`` sequential form_bonds calls starting from ``starting`` molecules.

    skip_hashes is copied internally so that filtering does not leak state back to the caller.
    """
    if steps == 0:
        return list(starting)

    working_hashes = set(skip_hashes) if skip_hashes is not None else set()
    frontier = list(starting)
    collected: List[yarpecule] = []

    for _ in range(steps):
        if not frontier:
            break
        next_frontier: List[yarpecule] = []
        for mol in frontier:
            new_products = list(yp.form_bonds([mol], hashes=working_hashes, inter=True, intra=True, hash_filter=True))
            next_frontier.extend(new_products)
        working_hashes.update(_.hash for _ in next_frontier)
        collected.extend(next_frontier)
        frontier = next_frontier
    return collected


def has_high_order_bond(bmat, max_order: int = MAX_BOND_ORDER) -> bool:
    """Return True if any off-diagonal bond order exceeds max_order."""
    size = len(bmat)
    for i in range(size):
        for j in range(i):
            if bmat[i, j] > max_order:
                return True
    return False


def enumerate_reactant(
    reactant: yarpecule,
    max_break: int = 3,
    max_form: int = 3,
    score_threshold: float = 0.0,
) -> Tuple[List[yarpecule], Dict[str, object]]:
    """
    Enumerate BNFM products for a single reactant.

    Returns a tuple of (new_products, record_dict) where new_products holds only
    structures that passed filtering and are unique with respect to this reactant.
    """
    # hash bookkeeping is local to this reactant to allow A->B and B->A to both be kept
    local_hashes: Set[float] = {reactant.hash}

    products: List[yarpecule] = []

    for n_break in range(max_break + 1):
        for n_form in range(max_form + 1):
            # break step (fresh each combo)
            if n_break == 0:
                break_products = [reactant]
            else:
                break_products = list(
                    yp.break_bonds(
                        [reactant],
                        n=n_break,
                        hashes=set(),
                        remove_redundant=True
                    )
                )

            if not break_products:
                continue

            # form step (fresh per combo)
            formed = _form_steps(
                break_products, n_form, skip_hashes=set()
            )
            for prod in formed:
                bmat = prod.bond_mats[0]
                if has_high_order_bond(bmat):
                    continue
                if prod.bond_mat_scores[0] > score_threshold:
                    continue
                if any(abs(chg) > 1 for chg in prod.fc):
                    continue
                if sum(1 for chg in prod.fc if chg != 0) > 2:
                    continue
                charged_atoms = [idx for idx, chg in enumerate(prod.fc) if chg != 0]
                if len(charged_atoms) == 2 and prod.adj_mat[charged_atoms[0], charged_atoms[1]] != 1:
                    continue
                if prod.hash in local_hashes:
                    continue
                local_hashes.add(prod.hash)
                products.append(prod)

    record = {
        "reactant": yarpecule_to_record(reactant),
        "products": [yarpecule_to_record(p) for p in products],
    }
    return products, record


def _enumerate_reactant_task(args: Tuple[yarpecule, int, int, float]):
    """Wrapper for process pool execution."""
    reactant, max_break, max_form, score_threshold = args
    return enumerate_reactant(
        reactant,
        max_break=max_break,
        max_form=max_form,
        score_threshold=score_threshold,
    )


def iterate_bnfm(
    reactant_smiles: Iterable[str],
    max_break: int = 3,
    max_form: int = 3,
    score_threshold: float = 0.0,
    output_path: str = "bnfm_results.pkl",
    log_file: str = "bnfm_iterator.log",
    max_workers: int = 1,
    retain_results: bool = False,
    start_time: Optional[float] = None,
) -> List[Dict[str, object]]:
    """
    Run iterative BNFM enumeration until no new species are found.

    Records are written to ``output_path`` in batches (one pickle dump per iteration)
    to avoid holding the full result set in memory. The output file is therefore a
    stream of pickled lists; use repeated pickle.load calls to consume it. Each record
    contains a reactant entry and its filtered, de-duplicated products (deduplication
    is local to each reactant; global hashes are only used for loop termination). Set
    retain_results=True to also accumulate and return the full list in memory.
    """
    configure_logging(log_file)
    run_start = time.time() if start_time is None else start_time

    # initialize reactants
    initial_reactants: List[yarpecule] = []
    global_hashes: Set[float] = set()
    for smi in reactant_smiles:
        mol = yp.yarpecule(smi)
        if mol.hash in global_hashes:
            continue
        global_hashes.add(mol.hash)
        initial_reactants.append(mol)

    results: List[Dict[str, object]] = [] if retain_results else []
    frontier = initial_reactants
    processed_reactants: Set[float] = set()
    iteration = 0

    with open(output_path, "wb") as output_file:
        while frontier:
            # skip any molecules that were already enumerated as reactants
            frontier = [r for r in frontier if r.hash not in processed_reactants]
            if not frontier:
                break

            iter_start = time.time()
            pre_species = len(global_hashes)
            worker_count = max(1, min(max_workers, len(frontier)))
            logger.info(
                "Iteration %d: starting enumeration with %d frontier reactants (%d known species total, %d workers)",
                iteration,
                len(frontier),
                pre_species,
                worker_count,
            )
            new_frontier: List[yarpecule] = []
            iteration_records: List[Dict[str, object]] = []
            task_args = [
                (reactant, max_break, max_form, score_threshold) for reactant in frontier
            ]
            total_frontier = len(frontier)
            completed = 0

            def consume_result(
                products: List[yarpecule], record: Dict[str, object], reactant_hash: float
            ) -> None:
                processed_reactants.add(reactant_hash)
                iteration_records.append(record)
                if retain_results:
                    results.append(record)
                for prod in products:
                    if prod.hash in global_hashes:
                        continue
                    global_hashes.add(prod.hash)
                    new_frontier.append(prod)
                products.clear()

            if worker_count > 1:
                with ProcessPoolExecutor(max_workers=worker_count) as executor:
                    future_map = {}
                    start_map = {}
                    for arg in task_args:
                        fut = executor.submit(_enumerate_reactant_task, arg)
                        future_map[fut] = arg[0]
                        start_map[fut] = time.time()
                    for fut in as_completed(future_map):
                        reactant = future_map[fut]
                        products, record = fut.result()
                        completed += 1
                        elapsed = time.time() - start_map[fut]
                        logger.info(
                            "Iteration %d progress %d/%d | reactant hash=%s | elapsed %.2fs",
                            iteration,
                            completed,
                            total_frontier,
                            reactant.hash,
                            elapsed,
                        )
                        consume_result(products, record, reactant.hash)
            else:
                for args in task_args:
                    reactant = args[0]
                    reactant_start = time.time()
                    products, record = _enumerate_reactant_task(args)
                    completed += 1
                    elapsed = time.time() - reactant_start
                    logger.info(
                        "Iteration %d progress %d/%d | reactant hash=%s | elapsed %.2fs",
                        iteration,
                        completed,
                        total_frontier,
                        reactant.hash,
                        elapsed,
                    )
                    consume_result(products, record, reactant.hash)

            if iteration_records:
                # Stream current iteration to disk to avoid holding all records in memory.
                pickle.dump(iteration_records, output_file, protocol=pickle.HIGHEST_PROTOCOL)
                output_file.flush()
                iteration_records.clear()

            added_species = len(global_hashes) - pre_species
            iter_elapsed = time.time() - iter_start
            logger.info(
                "Iteration %d: added %d new species | elapsed %.2fs",
                iteration,
                added_species,
                iter_elapsed,
            )

            if not new_frontier:
                break
            frontier = new_frontier
            iteration += 1

    total_elapsed = time.time() - run_start
    logger.info("Done! elapsed %.2fs", total_elapsed)

    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Iterative BNFM enumeration with hash-based deduplication."
    )
    parser.add_argument(
        "--smiles",
        nargs="+",
        required=True,
        help="One or more SMILES strings to use as starting reactants.",
    )
    parser.add_argument(
        "--max-break",
        type=int,
        default=3,
        help="Maximum number of bonds to break in a single enumeration step (inclusive).",
    )
    parser.add_argument(
        "--max-form",
        type=int,
        default=3,
        help="Maximum number of sequential bond formations to attempt (inclusive).",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.0,
        help="Keep products with bond_mat_scores[0] <= score_threshold.",
    )
    parser.add_argument(
        "--output",
        default="bnfm_results.pkl",
        help="Path to the pickle file where results are stored.",
    )
    parser.add_argument(
        "--log-file",
        default="bnfm_iterator.log",
        help="Path to write log output.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=54,
        help="Maximum parallel workers to enumerate frontier reactants within an iteration.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    iterate_bnfm(
        reactant_smiles=args.smiles,
        max_break=args.max_break,
        max_form=args.max_form,
        score_threshold=args.score_threshold,
        output_path=args.output,
        log_file=args.log_file,
        max_workers=args.max_workers,
        start_time=SCRIPT_START,
    )


if __name__ == "__main__":
    main()
