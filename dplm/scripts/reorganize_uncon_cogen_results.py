# NOTE: This file was created for this mplm_inference repository
"""
Reorganize DPLM2 unconditional cogeneration results to match our directory structure.

Current structure:
  length_x/
    aatype.fasta
    struct_token.fasta
    decoded_pdb/
      *.pdb

Target structure:
  length_x.fasta (moved from length_x/aatype.fasta)
  decoded_pdb/
    length_x/
      *.pdb (moved from length_x/decoded_pdb/)
"""

import os
import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Reorganize DPLM2 unconditional cogeneration results to match our structure")
    parser.add_argument("--source_root", type=str, help="Unconditional cogeneration output directory path containing length_x subdirectories")
    args = parser.parse_args()
    
    source_root = args.source_root
    
    decoded_pdb_root = os.path.join(source_root, "decoded_pdb")
    
    # Create decoded_pdb directory if it doesn't exist
    os.makedirs(decoded_pdb_root, exist_ok=True)
    
    # Get all length directories
    length_dirs = [d for d in os.listdir(source_root) 
                   if os.path.isdir(os.path.join(source_root, d)) and d.startswith("length_")]
    
    if not length_dirs:
        print(f"No length directories found in {source_root}")
        return
    
    print(f"Found {len(length_dirs)} length directories to reorganize\n")
    
    for length_dir in sorted(length_dirs):
        length_path = os.path.join(source_root, length_dir)
        
        # 1. Move aatype.fasta to root and rename
        aatype_src = os.path.join(length_path, "aatype.fasta")
        aatype_dst = os.path.join(source_root, f"{length_dir}.fasta")
        
        if os.path.exists(aatype_src):
            print(f"Processing {length_dir}...")
            shutil.move(aatype_src, aatype_dst)
            print(f"Moved aatype.fasta -> {length_dir}.fasta")
        else:
            print(f"Warning: {aatype_src} not found")
        
        # 2. Delete struct_token.fasta
        struct_token_src = os.path.join(length_path, "struct_token.fasta")
        if os.path.exists(struct_token_src):
            os.remove(struct_token_src)
            print(f"Deleted struct_token.fasta")
        
        # 3. Move all PDB files from decoded_pdb to decoded_pdb/length_x
        decoded_pdb_src = os.path.join(length_path, "decoded_pdb")
        decoded_pdb_dst = os.path.join(decoded_pdb_root, length_dir)
        
        if os.path.exists(decoded_pdb_src):
            # Create destination directory
            os.makedirs(decoded_pdb_dst, exist_ok=True)
            
            # Move all files from source to destination
            pdb_files = os.listdir(decoded_pdb_src)
            for pdb_file in pdb_files:
                src_file = os.path.join(decoded_pdb_src, pdb_file)
                dst_file = os.path.join(decoded_pdb_dst, pdb_file)
                shutil.move(src_file, dst_file)
            
            print(f"Moved {len(pdb_files)} PDB files to decoded_pdb/{length_dir}/")
            
            # Remove empty source directory
            try:
                os.rmdir(decoded_pdb_src)
            except OSError:
                pass
        else:
            print(f"Warning: {decoded_pdb_src} not found")
        
        # Remove empty length directory
        try:
            os.rmdir(length_path)
        except OSError as e:
            print(f"Warning: Could not remove {length_path}: {e}")
        
        print()
    
    print("Reorganization completed!")


if __name__ == "__main__":
    main()
