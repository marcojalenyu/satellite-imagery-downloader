import os

def rename_files(directory):
    try:
        # Get a list of files in the directory
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        
        # Sort files to maintain order during renaming (optional)
        files.sort()

        # Rename each file
        for i, filename in enumerate(files, start=1):
            # Get the full path of the current file
            old_path = os.path.join(directory, filename)
            
            # Construct the new filename
            new_filename = f"{i}_{filename}"
            new_path = os.path.join(directory, new_filename)

            # Rename the file
            os.rename(old_path, new_path)

        print(f"Successfully renamed {len(files)} files in {directory}.")
    except Exception as e:
        print(f"An error occurred: {e}")