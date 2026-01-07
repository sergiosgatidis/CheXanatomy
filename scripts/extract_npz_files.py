#!/usr/bin/env python3
"""
NPZ Extraction Script

Script to process multiple chest X-ray cases and create NPZ files for Paligemma generator
"""
import os
import argparse
from pathlib import Path
from anatomy_training_data import get_image_info


def extract_npz_files(input_dir: str, output_dir: str = "npz_files"):
    """
    Process multiple cases and create NPZ files for each
    
    Args:
        input_dir: Directory containing multiple case subdirectories
        output_dir: Directory to save the NPZ files
    """
    # Find all case directories
    case_dirs = []
    for item in os.listdir(input_dir):
        item_path = os.path.join(input_dir, item)
        if os.path.isdir(item_path):
            case_dirs.append(item_path)
    
    if not case_dirs:
        print(f"No case directories found in {input_dir}")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing {len(case_dirs)} cases...")
    print(f"Output directory: {output_dir}")
    
    # Detect orientation from input directory path
    orientation_suffix = ""
    if "PA" in input_dir.upper():
        orientation_suffix = "_PA"
    elif "LR" in input_dir.upper() or "LAT" in input_dir.upper():
        orientation_suffix = "_LR"
    
    if orientation_suffix:
        print(f"Detected orientation: {orientation_suffix.strip('_')}")
    
    successful_cases = 0
    failed_cases = 0
    
    # Process each case
    for case_dir in case_dirs:
        try:
            case_name = os.path.basename(case_dir)
            print(f"Processing case: {case_name}")
            
            # Define NPZ output path with orientation suffix
            npz_filename = f"{case_name}{orientation_suffix}.npz"
            npz_path = os.path.join(output_dir, npz_filename)
            
            # Get image info and export as NPZ
            image_info = get_image_info(case_dir, export_npz=True, npz_path=npz_path)
            
            print(f"  ✅ Saved {npz_filename} ({image_info['num_structures']} structures)")
            successful_cases += 1
            
        except Exception as e:
            print(f"  ❌ Error processing {case_dir}: {e}")
            failed_cases += 1
            continue
    
    print(f"\n🎉 Extraction complete!")
    print(f"📊 Successfully processed: {successful_cases} cases")
    if failed_cases > 0:
        print(f"❌ Failed: {failed_cases} cases")
    print(f"📁 NPZ files saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract NPZ files from chest X-ray case directories',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all cases in data/ directory
  python extract_npz_files.py --input_dir /path/to/data --output_dir npz_output
  
  # Process single case
  python extract_npz_files.py --single_case /path/to/case001 --output_dir npz_output
        """
    )
    
    parser.add_argument(
        '--input_dir', 
        required=False,
        help='Directory containing multiple case subdirectories'
    )
    parser.add_argument(
        '--output_dir', 
        default='npz_files', 
        help='Directory to save NPZ files (default: npz_files)'
    )
    parser.add_argument(
        '--single_case', 
        help='Process single case directory instead of multiple'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.single_case and not args.input_dir:
        parser.error("Either --input_dir or --single_case must be specified")
    
    if args.single_case and args.input_dir:
        parser.error("Cannot specify both --input_dir and --single_case")
    
    if args.single_case:
        # Process single case
        if not os.path.exists(args.single_case):
            print(f"Error: Case directory {args.single_case} does not exist")
            return 1
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Detect orientation from case path
        orientation_suffix = ""
        if "PA" in args.single_case.upper():
            orientation_suffix = "_PA"
        elif "LR" in args.single_case.upper() or "LAT" in args.single_case.upper():
            orientation_suffix = "_LR"
        
        case_name = os.path.basename(args.single_case)
        npz_filename = f"{case_name}{orientation_suffix}.npz"
        npz_path = os.path.join(args.output_dir, npz_filename)
        
        try:
            print(f"Processing single case: {case_name}")
            image_info = get_image_info(args.single_case, export_npz=True, npz_path=npz_path)
            print(f"✅ Saved {npz_filename} ({image_info['num_structures']} structures)")
            print(f"📁 NPZ file saved to: {npz_path}")
        except Exception as e:
            print(f"❌ Error processing {args.single_case}: {e}")
            return 1
            
    else:
        # Process multiple cases
        if not os.path.exists(args.input_dir):
            print(f"Error: Input directory {args.input_dir} does not exist")
            return 1
        
        extract_npz_files(args.input_dir, args.output_dir)
    
    return 0


if __name__ == "__main__":
    exit(main())