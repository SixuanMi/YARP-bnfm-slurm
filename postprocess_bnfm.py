"""
Post-process BNFM enumeration results.

Steps:
1. Flatten reactant -> products into reactant/product pairs.
2. Deduplicate reactions by unordered (reactant_hash, product_hash) so A->B and B->A are treated as duplicates.
3. Remove `geo` from all species records.
4. Generate plain and atom-mapped reaction SMILES (RDKit-based).
5. Save cleaned reactions to a pickle and print unique species/reaction counts.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import logging
import multiprocessing as mp
import os
import pickle
import time
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import numpy as np
from rdkit import Chem
from yarp.find_lewis import return_formals

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None

logger = logging.getLogger(__name__)


def return_smi(elements, bond_mat, fc):
    """
    Generate atom-mapped and plain SMILES from elements/bond_mat/fc.
    """
    try:
        mol = Chem.RWMol()

        # Add atoms with mapping numbers
        for i, element in enumerate(elements):
            atom = Chem.Atom(element)
            atom.SetAtomMapNum(i + 1)
            atom.SetFormalCharge(int(fc[i]))
            atom.SetNumRadicalElectrons(int(bond_mat[i, i] % 2))
            atom.SetNoImplicit(True)
            atom.SetNumExplicitHs(0)
            mol.AddAtom(atom)

        # Add bonds
        num_atoms = len(elements)
        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                order = int(round(bond_mat[i, j]))
                if order == 1:
                    bond_type = Chem.rdchem.BondType.SINGLE
                elif order == 2:
                    bond_type = Chem.rdchem.BondType.DOUBLE
                elif order == 3:
                    bond_type = Chem.rdchem.BondType.TRIPLE
                # No other bond types handled
                # If other types (e.g., aromatic, quadruple) are needed, add here
                else:
                    continue
                mol.AddBond(i, j, bond_type)

        mol.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(mol)

        mapped = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)

        mol_plain = Chem.RemoveHs(mol)
        for atom in mol_plain.GetAtoms():
            atom.SetAtomMapNum(0)
        plain = Chem.MolToSmiles(mol_plain, canonical=True, isomericSmiles=False)

        return mapped, plain
    except Exception as exc:
        print(f"Error generating SMILES: {exc}")
        return "", ""


def strip_geo(spec: Dict) -> Dict:
    out = dict(spec)
    out.pop("geo", None)
    # Replace q with fc derived from bond_mats/elements to avoid duplicating data.
    elements = [el.upper() for el in out["elements"]]
    bmat = _bond_mat_from_record(out["bond_mats"])
    out["fc"] = return_formals(bmat, elements)
    out.pop("q", None)
    return out


def reaction_key(h1: float, h2: float) -> Tuple[float, float]:
    return tuple(sorted((h1, h2)))


def _bond_mat_from_record(bond_mats) -> np.ndarray:
    bmat_raw = np.asarray(bond_mats)
    if bmat_raw.ndim == 3:
        return bmat_raw[0]
    elif bmat_raw.ndim == 2:
        return bmat_raw
    elif bmat_raw.ndim == 1:
        side = int(round(len(bmat_raw) ** 0.5))
        if side * side != len(bmat_raw):
            raise ValueError(f"Unable to reshape bond_mats of length {len(bmat_raw)} into square matrix.")
        return bmat_raw.reshape((side, side))
    else:
        raise ValueError(f"Unexpected bond_mats shape: {bmat_raw.shape}")


def species_smiles(spec: Dict) -> Tuple[str, str]:
    elements = [element.upper() for element in spec["elements"]]
    bmat = _bond_mat_from_record(spec["bond_mats"])
    fc = return_formals(bmat, elements)
    return return_smi(elements, bmat, fc)


def _species_smiles_from_parts(elements: Sequence[str], bond_mats) -> Tuple[str, str]:
    elements_up = [element.upper() for element in elements]
    bmat = _bond_mat_from_record(bond_mats)
    fc = return_formals(bmat, elements_up)
    return return_smi(elements_up, bmat, fc)


def _iter_bnfm_records(file_obj) -> Iterator[Dict]:
    """
    Yield BNFM reactant records from either:
    - a single pickled list
    - or a stream of pickled lists (one list per iteration)
    """
    while True:
        try:
            chunk = pickle.load(file_obj)
        except EOFError:
            break
        if isinstance(chunk, list):
            for entry in chunk:
                yield entry
        else:
            yield chunk


def _species_smiles_task(task: Tuple[float, Sequence[str], object]) -> Tuple[float, str, str, str]:
    """
    Compute SMILES for a single species record.

    Returns (hash, mapped, plain, error_message). error_message is empty string when successful.
    """
    species_hash, elements, bond_mats = task
    try:
        mapped, plain = _species_smiles_from_parts(elements, bond_mats)
        return species_hash, mapped, plain, ""
    except Exception as exc:
        return species_hash, "", "", str(exc)


def process(
    input_path: str,
    output_path: str,
    use_tqdm: bool = True,
    max_workers: int = 0,
    chunksize: int = 50,
) -> None:
    disable_tqdm = (not use_tqdm) or (tqdm is None)
    run_start = time.time()

    logger.info("Reading input pkl: %s", input_path)
    # Keep memory low: store one canonical record per species hash, and store reactions only as hash pairs.
    species_by_hash: Dict[float, Dict] = {}
    dedup_reactions: Dict[Tuple[float, float], Tuple[float, float]] = {}
    reactant_records = 0
    reaction_pairs = 0

    with open(input_path, "rb") as f:
        records: Iterable[Dict] = _iter_bnfm_records(f)
        if tqdm is not None:
            records = tqdm(records, desc="Reading/Dedup", unit="reactant", disable=disable_tqdm)
        for entry in records:
            reactant_records += 1
            reactant_raw = entry["reactant"]
            r_hash = reactant_raw["hash"]
            if r_hash not in species_by_hash:
                species_by_hash[r_hash] = strip_geo(reactant_raw)
            for product_raw in entry.get("products", []):
                reaction_pairs += 1
                p_hash = product_raw["hash"]
                if p_hash not in species_by_hash:
                    species_by_hash[p_hash] = strip_geo(product_raw)
                key = reaction_key(r_hash, p_hash)
                if key in dedup_reactions:
                    continue
                # Keep first-seen orientation (A>>B vs B>>A) while deduplicating by unordered pair.
                dedup_reactions[key] = (r_hash, p_hash)

    logger.info(
        "Read OK | reactants=%d | pairs=%d | unique_reactions=%d | unique_species=%d | elapsed %.2fs",
        reactant_records,
        reaction_pairs,
        len(dedup_reactions),
        len(species_by_hash),
        time.time() - run_start,
    )

    # Build cleaned reactions and species set
    reactions_out: List[Dict] = []
    failed_smiles = 0

    total_reactions = len(dedup_reactions)
    if total_reactions == 0:
        logger.warning("No reactions found; writing empty output.")
    else:
        if max_workers <= 0:
            max_workers = os.cpu_count() or 1
        max_workers = max(1, min(max_workers, max(1, len(species_by_hash))))

        # Generate SMILES per-species (NOT per-reaction) to avoid massive IPC and duplicated RDKit work.
        logger.info(
            "Generating species SMILES | species=%d | workers=%d | chunksize=%d",
            len(species_by_hash),
            max_workers,
            chunksize,
        )

        species_smiles: Dict[float, Tuple[str, str]] = {}
        species_tasks = (
            (spec_hash, spec["elements"], spec["bond_mats"]) for spec_hash, spec in species_by_hash.items()
        )

        if max_workers > 1:
            # Use spawn to avoid forking a huge parent process (common immediate OOM-kill on HPC).
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
                species_iter: Iterable[Tuple[float, str, str, str]] = executor.map(
                    _species_smiles_task, species_tasks, chunksize=max(1, chunksize)
                )
                if tqdm is not None:
                    species_iter = tqdm(
                        species_iter,
                        total=len(species_by_hash),
                        desc="Species SMILES",
                        unit="species",
                        disable=disable_tqdm,
                    )
                for spec_hash, mapped, plain, error in species_iter:
                    if error:
                        failed_smiles += 1
                        if failed_smiles <= 10:
                            logger.warning("SMILES failed for species %s: %s", spec_hash, error)
                    species_smiles[spec_hash] = (mapped, plain)
        else:
            species_iter2: Iterable[Tuple[float, str, str, str]] = (
                _species_smiles_task(task) for task in species_tasks
            )
            if tqdm is not None:
                species_iter2 = tqdm(
                    species_iter2,
                    total=len(species_by_hash),
                    desc="Species SMILES",
                    unit="species",
                    disable=disable_tqdm,
                )
            for spec_hash, mapped, plain, error in species_iter2:
                if error:
                    failed_smiles += 1
                    if failed_smiles <= 10:
                        logger.warning("SMILES failed for species %s: %s", spec_hash, error)
                species_smiles[spec_hash] = (mapped, plain)

        logger.info("Building reactions | reactions=%d", total_reactions)
        rxn_iter = dedup_reactions.values()
        if tqdm is not None:
            rxn_iter = tqdm(
                rxn_iter,
                total=total_reactions,
                desc="Building reactions",
                unit="reaction",
                disable=disable_tqdm,
            )

        species_used: Dict[float, Dict] = {}
        for r_hash, p_hash in rxn_iter:
            r_spec = species_by_hash[r_hash]
            p_spec = species_by_hash[p_hash]
            species_used[r_hash] = r_spec
            species_used[p_hash] = p_spec
            r_mapped, r_plain = species_smiles.get(r_hash, ("", ""))
            p_mapped, p_plain = species_smiles.get(p_hash, ("", ""))
            reactions_out.append(
                {
                    "reactant": r_spec,
                    "product": p_spec,
                    "reaction_smiles": f"{r_plain}>>{p_plain}",
                    "reaction_smiles_mapped": f"{r_mapped}>>{p_mapped}",
                }
            )

        logger.info("Unique species: %d", len(species_used))
        logger.info("Unique reactions: %d", len(reactions_out))
        if failed_smiles:
            logger.info("SMILES failures: %d (first 10 shown above)", failed_smiles)

    logger.info("Writing output pkl: %s", output_path)
    with open(output_path, "wb") as f:
        pickle.dump(reactions_out, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Write OK | elapsed %.2fs", time.time() - run_start)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-process BNFM pkl to deduplicate and annotate reactions.")
    parser.add_argument("--input", required=True, help="Input pkl from bnfm_iterator.")
    parser.add_argument(
        "--output", default="bnfm_cleaned.pkl", help="Output pkl path for deduplicated reactions."
    )
    parser.add_argument(
        "--tqdm",
        action="store_true",
        help="Enable tqdm progress bars (requires tqdm). Disabled by default.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        # default=0,
        default=8,
        help="Parallel workers for SMILES generation (0=auto). Use 1 to disable parallelism.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=50,
        help="ProcessPoolExecutor chunksize for SMILES generation (default: 50).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    use_tqdm = bool(args.tqdm) and (tqdm is not None)
    if args.tqdm and (tqdm is None):
        logger.warning("--tqdm requested but tqdm is not installed; continuing without progress bars.")

    process(
        args.input,
        args.output,
        use_tqdm=use_tqdm,
        max_workers=args.max_workers,
        chunksize=args.chunksize,
    )


if __name__ == "__main__":
    main()
