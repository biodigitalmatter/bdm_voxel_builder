import os


def get_nth_newest_file_in_folder(folder_path, n):
    try:
        # Get a list of files in the folder
        files = [
            os.path.join(folder_path, filename) for filename in os.listdir(folder_path)
        ]

        # Sort the files by change time (modification time) in descending order
        files.sort(key=lambda x: os.path.getmtime(x), reverse=True)

        # Return the newest file
        if files:
            return files[min(n, len(files))]
        else:
            print("Folder is empty.")
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None
