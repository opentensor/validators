from huggingface_hub import HfFileSystem


def check_file_exists(filepath: str) -> bool:
    """Checks if file exists in Hugging Face Hub
    Args:
        filepath (str): File path
    Returns:
        bool: True if file exists, False otherwise
    """
    hf_file_system = HfFileSystem()
    file_exists = len(hf_file_system.glob(filepath)) != 0

    return file_exists
