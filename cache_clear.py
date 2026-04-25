import os
from pathlib import Path

def clear_all_caches():
    """
    Finds and deletes all generated cache files (.pt) and the class map (.json)
    from the FeGAN project data directory.
    """
    # --- Configuration ---
    # The root directory where all your farm data is stored.
    root_path = Path("C:/Users/kusha/FeGAN_Project/data/main_data")

    print(f"--- Starting Cache Deletion in: {root_path} ---\n")

    # 1. Define all cache files to be deleted
    # The main validation cache created by the server
    validation_cache = root_path / "validation_graph_cache.pt"
    
    # The universal class map created by the server
    class_map_file = root_path / "class_map.json"
    
    # A list of all potential cache files to check and delete
    files_to_delete = [validation_cache, class_map_file]

    # Find all farm-specific caches (e.g., "Farm_1_graph_cache.pt")
    farm_caches = root_path.glob('Farm_*/*_graph_cache.pt')
    files_to_delete.extend(farm_caches)

    # 2. Iterate and delete files if they exist
    deleted_count = 0
    for file_path in files_to_delete:
        if file_path.exists():
            try:
                os.remove(file_path)
                print(f"DELETED: {file_path}")
                deleted_count += 1
            except OSError as e:
                print(f"Error deleting {file_path}: {e}")
        else:
            # This is normal if the file was never created
            pass
            
    if deleted_count == 0:
        print("No cache files found to delete. The directory is already clean.")
    else:
        print(f"\nSuccessfully deleted {deleted_count} cache file(s).")
        
    print("\n--- Cache Deletion Complete ---")


if __name__ == "__main__":
    clear_all_caches()
