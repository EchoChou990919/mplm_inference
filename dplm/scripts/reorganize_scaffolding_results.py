# NOTE: This file was created for this mplm_inference repository
"""
Reorganize DPLM2 results to match our directory structure.

Current structure:
  scaffold_fasta/
    case_name/
      aatype.fasta
      decoded_pdb/
        *.pdb

Target structure:
  case_name.fasta (moved from scaffold_fasta/case_name/aatype.fasta)
  decoded_pdb/
    case_name/
      *.pdb (moved from scaffold_fasta/case_name/decoded_pdb/)
"""

import os
import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Reorganize DPLM2 results to match our structure")
    parser.add_argument("--source_root", type=str, help="Motif-Scaffolding output directory path containing scaffold_fasta subdirectory")
    args = parser.parse_args()
    
    source_root = args.source_root
    
    scaffold_fasta_dir = os.path.join(source_root, "scaffold_fasta")
    decoded_pdb_root = os.path.join(source_root, "decoded_pdb")
    
    # Create decoded_pdb directory if it doesn't exist
    os.makedirs(decoded_pdb_root, exist_ok=True)
    
    # Get all case directories in scaffold_fasta
    case_dirs = [d for d in os.listdir(scaffold_fasta_dir) 
                 if os.path.isdir(os.path.join(scaffold_fasta_dir, d))]
    
    if not case_dirs:
        print(f"No case directories found in {scaffold_fasta_dir}")
        return
    
    print(f"Found {len(case_dirs)} case directories to reorganize\n")
    
    for case_name in sorted(case_dirs):
        case_path = os.path.join(scaffold_fasta_dir, case_name)
        
        # 1. Move aatype.fasta to root and rename
        aatype_src = os.path.join(case_path, "aatype.fasta")
        aatype_dst = os.path.join(source_root, f"{case_name}.fasta")
        
        if os.path.exists(aatype_src):
            print(f"Processing {case_name}...")
            shutil.move(aatype_src, aatype_dst)
            print(f"Moved aatype.fasta -> {case_name}.fasta")
        else:
            print(f"Warning: {aatype_src} not found")
        
        # 2. Move all PDB files from decoded_pdb to decoded_pdb/case_name
        decoded_pdb_src = os.path.join(case_path, "decoded_pdb")
        decoded_pdb_dst = os.path.join(decoded_pdb_root, case_name)
        
        if os.path.exists(decoded_pdb_src):
            # Create destination directory
            os.makedirs(decoded_pdb_dst, exist_ok=True)
            
            # Move all files from source to destination
            pdb_files = os.listdir(decoded_pdb_src)
            for pdb_file in pdb_files:
                src_file = os.path.join(decoded_pdb_src, pdb_file)
                dst_file = os.path.join(decoded_pdb_dst, pdb_file)
                shutil.move(src_file, dst_file)
            
            print(f"Moved {len(pdb_files)} PDB files to decoded_pdb/{case_name}/")
            
            # Remove empty source directory
            try:
                os.rmdir(decoded_pdb_src)
            except OSError:
                pass
        else:
            print(f"Warning: {decoded_pdb_src} not found")
        
        # Remove empty case directory from scaffold_fasta
        try:
            os.rmdir(case_path)
        except OSError as e:
            print(f"Warning: Could not remove {case_path}: {e}")
        
        print()
    
    # Remove empty scaffold_fasta directory if it's empty
    try:
        os.rmdir(scaffold_fasta_dir)
        print("Removed empty scaffold_fasta directory")
    except OSError:
        remaining = os.listdir(scaffold_fasta_dir)
        if remaining:
            print(f"Warning: scaffold_fasta directory still has contents: {remaining}")
    
    print("Reorganization completed!")


if __name__ == "__main__":
    main()
