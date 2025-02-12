from pynput.keyboard import Key, Controller

keyboard = Controller()

def PressKey(hexKeyCode):
    """
    Simulates a key press using the provided hex key code.
    """
    # Convert the hex key code to its character equivalent
    key = chr(hexKeyCode)
    keyboard.press(key)

def ReleaseKey(hexKeyCode):
    """
    Simulates a key release using the provided hex key code.
    """
    # Convert the hex key code to its character equivalent
    key = chr(hexKeyCode)
    keyboard.release(key)
