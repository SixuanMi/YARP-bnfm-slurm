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
import multiprocessing as mp
import os
import pickle
import sys
import time
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import yarp as yp
from yarp.yarpecule import yarpecule

MAX_BOND_ORDER = 3
SCRIPT_START = time.time()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None


def _format_bytes(num_bytes: Optional[float]) -> str:
    if num_bytes is None:
        return "NA"
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    value = float(num_bytes)
    for unit in units:
        if abs(value) < 1024.0:
            return f"{value:,.2f}{unit}"
        value /= 1024.0
    return f"{value:,.2f}EB"


def _current_rss_bytes(include_children: bool = True) -> Optional[int]:
    # Prefer psutil when available so we can optionally include child processes.
    if psutil is not None:
        try:
            proc = psutil.Process(os.getpid())
            rss = int(proc.memory_info().rss)
            if include_children:
                for child in proc.children(recursive=True):
                    try:
                        rss += int(child.memory_info().rss)
                    except Exception:
                        continue
            return rss
        except Exception:
            pass

    # Linux fallback: /proc/self/statm provides current RSS in pages.
    try:
        with open("/proc/self/statm") as f:
            parts = f.readline().split()
        if len(parts) >= 2:
            rss_pages = int(parts[1])
            page_size = os.sysconf("SC_PAGE_SIZE")
            return rss_pages * page_size
    except Exception:
        pass
    return None


def _estimate_set_of_floats_bytes(values: Set[float]) -> int:
    # sys.getsizeof includes the set table but not the referenced float objects.
    return sys.getsizeof(values) + len(values) * sys.getsizeof(0.0)


def _estimate_species_record_bytes(record: Dict[str, object]) -> int:
    # Best-effort estimate for yarpecule_to_record output (dict of small items and numpy arrays).
    size = sys.getsizeof(record)
    for key in ("adj_mat", "elements", "q", "bond_mats", "bond_mat_scores", "hash"):
        val = record.get(key)
        if val is None:
            continue
        try:
            size += sys.getsizeof(val)
        except TypeError:
            continue
    return size


def _estimate_record_bytes(record: Dict[str, object], product_sample: int = 3) -> int:
    size = sys.getsizeof(record)
    reactant = record.get("reactant")
    if isinstance(reactant, dict):
        size += _estimate_species_record_bytes(reactant)
    products = record.get("products", [])
    try:
        size += sys.getsizeof(products)
    except TypeError:
        products = []
    if isinstance(products, list) and products:
        sample = products[: min(product_sample, len(products))]
        # Many product records share a common schema; sample a few for a stable estimate.
        avg = sum(_estimate_species_record_bytes(p) for p in sample if isinstance(p, dict)) / max(len(sample), 1)
        size += int(avg * len(products))
    return int(size)


def _estimate_iteration_records_bytes(iteration_records: List[Dict[str, object]]) -> int:
    size = sys.getsizeof(iteration_records)
    if not iteration_records:
        return size
    sample = iteration_records[-min(3, len(iteration_records)) :]
    avg = sum(_estimate_record_bytes(r) for r in sample) / len(sample)
    return int(size + avg * len(iteration_records))


def _estimate_yarpecule_bytes(mol: yarpecule) -> int:
    # Best-effort estimate focusing on large numpy arrays.
    size = sys.getsizeof(mol)
    for attr in ("adj_mat", "geo", "atom_hashes"):
        val = getattr(mol, attr, None)
        if val is None:
            continue
        try:
            size += sys.getsizeof(val)
        except TypeError:
            continue
    bond_mats = getattr(mol, "bond_mats", None)
    if isinstance(bond_mats, list):
        size += sys.getsizeof(bond_mats)
        for b in bond_mats:
            try:
                size += sys.getsizeof(b)
            except TypeError:
                continue
    return int(size)


def _estimate_new_frontier_bytes(new_frontier: List[yarpecule]) -> int:
    size = sys.getsizeof(new_frontier)
    if not new_frontier:
        return size
    sample = new_frontier[-min(3, len(new_frontier)) :]
    avg = sum(_estimate_yarpecule_bytes(m) for m in sample) / len(sample)
    return int(size + avg * len(new_frontier))


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
) -> Iterable[yarpecule]:
    """
    Perform ``steps`` sequential form_bonds calls starting from ``starting`` molecules.

    This streams products as they are generated to reduce peak memory versus holding
    every intermediate in a list. skip_hashes is copied internally so that filtering
    does not leak state back to the caller.
    """
    if steps == 0:
        # Yield instead of allocating a copy to keep memory flat when max_form is large.
        for mol in starting:
            yield mol
        return

    working_hashes = set(skip_hashes) if skip_hashes is not None else set()
    frontier = list(starting)

    for _ in range(steps):
        if not frontier:
            break
        next_frontier: List[yarpecule] = []
        step_hashes: List[float] = []
        for mol in frontier:
            for prod in yp.form_bonds(
                [mol],
                hashes=working_hashes,
                inter=True,
                intra=True,
                hash_filter=True,
            ):
                next_frontier.append(prod)
                step_hashes.append(prod.hash)
                yield prod
        # keep form_bonds hash filtering behavior consistent while avoiding an extra copy
        working_hashes.update(step_hashes)
        frontier = next_frontier


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
    batch_size: int = 0,
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
        # Compute the break step once per n_break; reuse for all n_form to avoid redundant work.
        if n_break == 0:
            base_break_products = [reactant]
        else:
            base_break_products = list(
                yp.break_bonds(
                    [reactant],
                    n=n_break,
                    hashes=set(),
                    remove_redundant=True
                )
            )

        if not base_break_products:
            continue

        for n_form in range(max_form + 1):
            # form step (fresh per combo); optionally chunk break products to cap peak memory
            if batch_size and batch_size > 0:
                batches = (
                    base_break_products[i : i + batch_size]
                    for i in range(0, len(base_break_products), batch_size)
                )
            else:
                batches = (base_break_products,)

            for break_batch in batches:
                formed = _form_steps(
                    break_batch, n_form, skip_hashes=set()
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


def _enumerate_bnfm_combo_task(args: Tuple[yarpecule, int, int, float, int]) -> List[yarpecule]:
    """
    Enumerate a single (n_break, n_form) combination for one reactant.

    Returns only the products for this combo; caller is responsible for per-reactant
    aggregation and cross-combo de-duplication.
    """
    reactant, n_break, n_form, score_threshold, batch_size = args
    local_hashes: Set[float] = {reactant.hash}
    products: List[yarpecule] = []

    # Stream break products to avoid holding a large intermediate list.
    chunk_size = batch_size if batch_size and batch_size > 0 else 1
    buffer: List[yarpecule] = []

    def _process_batch(batch: List[yarpecule]) -> None:
        nonlocal products
        if not batch:
            return
        formed = _form_steps(batch, n_form, skip_hashes=set())
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

    if n_break == 0:
        buffer.append(reactant)
        _process_batch(buffer)
        buffer.clear()
    else:
        for mid in yp.break_bonds(
            [reactant],
            n=n_break,
            hashes=set(),
            remove_redundant=True
        ):
            buffer.append(mid)
            if len(buffer) >= chunk_size:
                _process_batch(buffer)
                buffer.clear()
        if buffer:
            _process_batch(buffer)
            buffer.clear()

    return products


def _enumerate_bnfm_combo_task_with_meta(
    args: Tuple[yarpecule, int, int, float, int]
) -> Tuple[Tuple[yarpecule, int, int, float, int], List[yarpecule]]:
    """Pool-safe wrapper that returns both the input args and products."""
    return args, _enumerate_bnfm_combo_task(args)


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
    log_memory: bool = False,
    log_memory_every: int = 1,
    mp_start_method: Optional[str] = None,
    batch_size: int = 0,
    max_tasks_per_child: int = 0,
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

    results: List[Dict[str, object]] = [] if retain_results else [] # 这里不论如何都是 []
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
            iteration_products_total = 0
            reactant_states: Dict[float, Dict[str, object]] = {}
            task_args: List[Tuple[yarpecule, int, int, float, int]] = []
            combos_per_reactant = (max_break + 1) * (max_form + 1)

            for reactant in frontier:
                reactant_states[reactant.hash] = {
                    "reactant": reactant,
                    "products": [],
                    "local_hashes": {reactant.hash},
                    "remaining": combos_per_reactant,
                }
                for n_break in range(max_break + 1):
                    for n_form in range(max_form + 1):
                        task_args.append((reactant, n_break, n_form, score_threshold, batch_size))

            total_tasks = len(task_args)
            completed_tasks = 0

            def consume_result(
                products: List[yarpecule], reactant_hash: float
            ) -> None:
                nonlocal iteration_products_total
                state = reactant_states[reactant_hash]
                state_hashes: Set[float] = state["local_hashes"]  # type: ignore
                state_products: List[yarpecule] = state["products"]  # type: ignore

                for prod in products:
                    if prod.hash in state_hashes:
                        continue
                    state_hashes.add(prod.hash)
                    state_products.append(prod)
                    if prod.hash in global_hashes:
                        continue
                    global_hashes.add(prod.hash)
                    new_frontier.append(prod)
                products.clear()

                state["remaining"] -= 1
                if state["remaining"] == 0:
                    processed_reactants.add(reactant_hash)
                    record_products = [yarpecule_to_record(p) for p in state_products]
                    record = {
                        "reactant": yarpecule_to_record(state["reactant"]),  # type: ignore
                        "products": record_products,
                    }
                    iteration_records.append(record)
                    iteration_products_total += len(record_products)
                    if retain_results:
                        results.append(record)
                    state_products.clear()
                    state_hashes.clear()
                    state["reactant"] = None  # type: ignore

            def log_memory_state() -> None:
                if not log_memory:
                    return
                if log_memory_every > 1 and (completed_tasks % log_memory_every) != 0 and completed_tasks != total_tasks:
                    return
                rss_self = _current_rss_bytes(include_children=False)
                rss_total = _current_rss_bytes(include_children=True) if psutil is not None else rss_self
                logger.info(
                    "Iteration %d memory | rss=%s (total=%s) | global_hashes=%d (%s) | processed=%d (%s) | iteration_records=%d (%s) | new_frontier=%d (%s) | iter_products=%d",
                    iteration,
                    _format_bytes(rss_self),
                    _format_bytes(rss_total),
                    len(global_hashes),
                    _format_bytes(_estimate_set_of_floats_bytes(global_hashes)),
                    len(processed_reactants),
                    _format_bytes(_estimate_set_of_floats_bytes(processed_reactants)),
                    len(iteration_records),
                    _format_bytes(_estimate_iteration_records_bytes(iteration_records)),
                    len(new_frontier),
                    _format_bytes(_estimate_new_frontier_bytes(new_frontier)),
                    iteration_products_total,
                )

            if worker_count > 1:
                # NOTE: On Linux the default start method is usually "fork". For large parent processes
                # (big frontier/records), forking can trigger copy-on-write memory amplification.
                # Users can pass mp_start_method="spawn" (or "forkserver") to avoid forking the parent.
                mp_context = mp.get_context(mp_start_method) if mp_start_method else mp.get_context()
                tasks_per_child = max_tasks_per_child if max_tasks_per_child and max_tasks_per_child > 0 else 1
                with mp_context.Pool(processes=worker_count, maxtasksperchild=tasks_per_child) as pool:
                    start_times: Dict[Tuple[yarpecule, int, int, float, int], float] = {}
                    for arg in task_args:
                        start_times[arg] = time.time()
                    for arg, products in pool.imap_unordered(
                        _enumerate_bnfm_combo_task_with_meta, task_args, chunksize=1
                    ):
                        reactant_obj, n_break, n_form, _, _ = arg
                        completed_tasks += 1
                        elapsed = time.time() - start_times[arg]
                        consume_result(products, reactant_obj.hash)
                        logger.info(
                            "Iteration %d progress %d/%d combos | reactant hash=%s | b%d f%d | elapsed %.2fs",
                            iteration,
                            completed_tasks,
                            total_tasks,
                            reactant_obj.hash,
                            n_break,
                            n_form,
                            elapsed,
                        )
                        log_memory_state()
            else:
                for args in task_args:
                    reactant_obj, n_break, n_form, _, _ = args
                    reactant_start = time.time()
                    products = _enumerate_bnfm_combo_task(args)
                    completed_tasks += 1
                    elapsed = time.time() - reactant_start
                    consume_result(products, reactant_obj.hash)
                    logger.info(
                        "Iteration %d progress %d/%d combos | reactant hash=%s | b%d f%d | elapsed %.2fs",
                        iteration,
                        completed_tasks,
                        total_tasks,
                        reactant_obj.hash,
                        n_break,
                        n_form,
                        elapsed,
                    )
                    log_memory_state()

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
    parser.add_argument(
        "--log-memory",
        action="store_true",
        help="Log approximate memory usage of key containers after each completed reactant.",
    )
    parser.add_argument(
        "--log-memory-every",
        type=int,
        default=1,
        help="When --log-memory is enabled, log memory every N completed reactants (default: 1).",
    )
    parser.add_argument(
        "--mp-start-method",
        default="",
        help='Multiprocessing start method for parallel mode: "spawn", "fork", or "forkserver" (default: system).',
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        # default=0,
        default=20,
        help="When >0, process break-step products in chunks of this size during form steps to cap per-task memory (default: 0 = all at once).",
    )
    parser.add_argument(
        "--max-tasks-per-child",
        type=int,
        # default=0,
        default=1,
        help="When >0, restart worker processes after this many tasks to release memory (uses ProcessPoolExecutor max_tasks_per_child; default: 0 = never).",
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
        log_memory=args.log_memory,
        log_memory_every=args.log_memory_every,
        mp_start_method=(args.mp_start_method or None),
        batch_size=args.batch_size,
        max_tasks_per_child=args.max_tasks_per_child,
    )


if __name__ == "__main__":
    main()
