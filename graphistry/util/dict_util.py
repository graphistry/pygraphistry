def assign(original, updates):
    """
    returns a new dict with original keys and updated values
    """
    return {
        k: updates[k] if k in updates else v for k, v in original.items()
    }
