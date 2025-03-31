import os
import shutil

def aggregate(root_dir):

    """
    Aggregates all __pycache__ directories in the project (except the one at the root)
    into a single __pycache__ directory at the root of the project.
    
    Each file is renamed to include its original relative path (with underscores)
    to avoid name conflicts.
    """

    # Destination directory at the root
    dest_dir = os.path.join(root_dir, "__pycache__")
    
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    # Walk the directory tree starting from root_dir
    for dirpath, dirnames, filenames in os.walk(root_dir):

        # Skip the destination directory itself
        if os.path.abspath(dirpath) == os.path.abspath(dest_dir):
            continue

        # Process directories named "__pycache__"
        if "__pycache__" in dirnames:

            cache_dir = os.path.join(dirpath, "__pycache__")

            for file in os.listdir(cache_dir):

                src_file = os.path.join(cache_dir, file)

                # Compute relative path from root and replace path separators with underscores
                rel_path = os.path.relpath(dirpath, root_dir)
                new_filename = rel_path.replace(os.sep, "_") + "_" + file
                dest_file = os.path.join(dest_dir, new_filename)

                # Copy file to destination
                shutil.copy2(src_file, dest_file)

            # Optionally, remove the original __pycache__ directory:

            # shutil.rmtree(cache_dir)
