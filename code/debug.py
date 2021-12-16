is_debug_mode = True
is_dev_mode = True
def debug_print(something = ''):
  if is_debug_mode:
    print(something)

def get_debug_mode():
  return is_debug_mode

def get_dev_mode():
  return is_dev_mode
