from asyncio.windows_events import NULL


def validateThreshold(str):
    castedInt = int(str,10)
    if(castedInt < 0 or castedInt > 30):
      return False
    return castedInt
