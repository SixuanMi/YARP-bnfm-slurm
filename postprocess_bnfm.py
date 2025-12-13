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
import pickle
from typing import Dict, List, Tuple

import numpy as np
from rdkit import Chem
from yarp.find_lewis import return_formals


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
    return out


def reaction_key(h1: float, h2: float) -> Tuple[float, float]:
    return tuple(sorted((h1, h2)))


def species_smiles(spec: Dict) -> Tuple[str, str]:
    elements = [element.upper() for element in spec["elements"]]
    bmat_raw = np.array(spec["bond_mats"])
    if bmat_raw.ndim == 3:
        bmat = bmat_raw[0]
    elif bmat_raw.ndim == 2:
        bmat = bmat_raw
    elif bmat_raw.ndim == 1:
        side = int(round(len(bmat_raw) ** 0.5))
        if side * side != len(bmat_raw):
            raise ValueError(f"Unable to reshape bond_mats of length {len(bmat_raw)} into square matrix.")
        bmat = bmat_raw.reshape((side, side))
    else:
        raise ValueError(f"Unexpected bond_mats shape: {bmat_raw.shape}")
    fc = return_formals(bmat, elements)
    return return_smi(elements, bmat, fc)


def process(input_path: str, output_path: str) -> None:
    # Support both single-list pickle and streamed pickles (one list per iteration)
    raw = []
    with open(input_path, "rb") as f:
        try:
            first = pickle.load(f)
        except EOFError:
            first = []
        if isinstance(first, list):
            raw.extend(first)
        else:
            raw.append(first)
        # Read any additional pickled chunks (for streamed outputs)
        while True:
            try:
                chunk = pickle.load(f)
            except EOFError:
                break
            if isinstance(chunk, list):
                raw.extend(chunk)
            else:
                raw.append(chunk)

    dedup_reactions = {}
    for entry in raw:
        reactant = entry["reactant"]
        for product in entry.get("products", []):
            key = reaction_key(reactant["hash"], product["hash"])
            if key in dedup_reactions:
                continue
            dedup_reactions[key] = (reactant, product)

    # Build cleaned reactions and species set
    reactions_out = []
    species_seen = {}

    for (r_hash, p_hash), (r_spec, p_spec) in dedup_reactions.items():
        r_clean = strip_geo(r_spec)
        p_clean = strip_geo(p_spec)
        species_seen[r_clean["hash"]] = r_clean
        species_seen[p_clean["hash"]] = p_clean

        r_mapped, r_plain = species_smiles(r_clean)
        p_mapped, p_plain = species_smiles(p_clean)
        reactions_out.append(
            {
                "reactant": r_clean,
                "product": p_clean,
                "reaction_smiles": f"{r_plain}>>{p_plain}",
                "reaction_smiles_mapped": f"{r_mapped}>>{p_mapped}",
            }
        )

    print(f"Unique species: {len(species_seen)}")
    print(f"Unique reactions: {len(reactions_out)}")

    with open(output_path, "wb") as f:
        pickle.dump(reactions_out, f)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-process BNFM pkl to deduplicate and annotate reactions.")
    parser.add_argument("--input", required=True, help="Input pkl from bnfm_iterator.")
    parser.add_argument(
        "--output", default="bnfm_cleaned.pkl", help="Output pkl path for deduplicated reactions."
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    process(args.input, args.output)


if __name__ == "__main__":
    main()
