import hashlib
import os
import sys

def calculate_sha256(file_path, buffer_size=65536):
    """Calculate SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    try:
        with open(file_path, 'rb') as f:
            while True:
                data = f.read(buffer_size)
                if not data:
                    break
                sha256.update(data)
        return sha256.hexdigest()
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def verify_sha256_checksums(checksum_file='SHA256SUMS.txt'):
    """Verify files against SHA256 checksums from a checksum file."""
    verified_files = []
    failed_files = []
    missing_files = []
    error_files = []
    
    # Read the checksum file
    try:
        with open(checksum_file, 'r') as f:
            checksum_lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: {checksum_file} not found!")
        return [], [], [], []
    
    print("Starting SHA256 verification...")
    print("=" * 80)
    
    data_dir = os.path.dirname(checksum_file) or 'data'
    
    for line_num, line in enumerate(checksum_lines, 1):
        line = line.strip()
        if not line:
            continue
            
        # Parse the checksum and filename
        parts = line.split()
        if len(parts) < 2:
            print(f"Warning: Malformed line {line_num}: {line}")
            continue
            
        expected_hash = parts[0]
        filename = ' '.join(parts[1:])  # Handle filenames with spaces
        file_path = os.path.join(data_dir, filename)
        
        # Check if file exists
        if not os.path.exists(file_path):
            missing_files.append(filename)
            print(f"❌ MISSING: {filename}")
            continue
        
        # Calculate actual hash of the file
        actual_hash = calculate_sha256(file_path)
        
        if actual_hash is None:
            error_files.append(filename)
            print(f"❌ ERROR: {filename} - Could not read file")
            continue
        
        # Compare hashes
        if actual_hash == expected_hash:
            verified_files.append(filename)
            print(f"✅ VERIFIED: {filename}")
        else:
            failed_files.append(filename)
            print(f"❌ FAILED: {filename}")
            print(f"   Expected: {expected_hash}")
            print(f"   Actual:   {actual_hash}")
    
    # Print summary
    print("=" * 80)
    print("\n📊 VERIFICATION SUMMARY:")
    print(f"✅ Verified: {len(verified_files)} files")
    print(f"❌ Failed: {len(failed_files)} files")
    print(f"⚠️  Missing: {len(missing_files)} files")
    print(f"🚫 Errors: {len(error_files)} files")
    print(f"📋 Total files in checksum: {len(checksum_lines)}")
    
    if verified_files:
        print(f"\n✅ Verified files:")
        for file in verified_files:
            print(f"  - {file}")
    
    if failed_files:
        print(f"\n❌ Failed files:")
        for file in failed_files:
            print(f"  - {file}")
    
    if missing_files:
        print(f"\n⚠️  Missing files:")
        for file in missing_files:
            print(f"  - {file}")
    
    if error_files:
        print(f"\n🚫 Files with errors:")
        for file in error_files:
            print(f"  - {file}")
    
    return verified_files, failed_files, missing_files, error_files

def main():
    """Main function to run the verification."""
    # Get the project root (parent of src)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    checksum_file = os.path.join(project_root, 'data', 'SHA256SUMS.txt')
    
    if not os.path.exists(checksum_file):
        print(f"Error: {checksum_file} not found!")
        print("Please make sure the SHA256SUMS.txt file is in the 'data/' directory.")
        return
    
    print(f"Using checksum file: {checksum_file}")
    print(f"Current directory: {os.getcwd()}")
    print()
    
    verified, failed, missing, errors = verify_sha256_checksums(checksum_file)
    
    # Return appropriate exit code
    if failed or missing or errors:
        print("\n❌ Verification completed with issues!")
        sys.exit(1)
    else:
        print("\n🎉 All files verified successfully!")
        sys.exit(0)

if __name__ == "__main__":
    main()