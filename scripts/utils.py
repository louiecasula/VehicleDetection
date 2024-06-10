from asyncio.windows_events import NULL

def validateThreshold(str):
    casted_int = int(str, 10)  # Convert the string to an integer
    if 0 <= casted_int <= 30:  # Check if the integer is within the valid range
      return casted_int
    return False
