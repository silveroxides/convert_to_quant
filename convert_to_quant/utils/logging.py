
import logging
import sys
import functools

# Custom Levels
# MINIMAL (30): WARNING+ (Reduced)
# NORMAL (20): INFO+ (Default)
# VERBOSE (15): Custom+ (Increased)
# DEBUG (10): DEBUG+ (Every function call)

MINIMAL_LEVEL = 30 # Use logging.WARNING
NORMAL_LEVEL = 20  # Use logging.INFO
VERBOSE_LEVEL = 15 # Custom level between INFO and DEBUG
DEBUG_LEVEL = 10   # logging.DEBUG

logging.addLevelName(VERBOSE_LEVEL, "VERBOSE")
logging.addLevelName(MINIMAL_LEVEL, "MINIMAL") # Alias for warning basically, or custom? 
# Actually if MINIMAL is just WARNING, we can just use WARNING. 
# But let's define specific levels to be clear.

class CustomFormatter(logging.Formatter):
    def format(self, record):
        if record.levelno <= DEBUG_LEVEL:
             # Debug: Full trace info
            self._style._fmt = "[%(levelname)s] %(filename)s:%(lineno)d - %(message)s"
        elif record.levelno <= VERBOSE_LEVEL:
            # Verbose: Detail
            self._style._fmt = "%(message)s"
        elif record.levelno <= NORMAL_LEVEL:
            # Normal: Standard output
            self._style._fmt = "%(message)s"
        else:
            # Minimal/Warning
            self._style._fmt = "[%(levelname)s] %(message)s"
            
        return super().format(record)

def setup_logging(verbose_arg: str = "NORMAL"):
    """
    Setup logging based on verbosity name.
    """
    level_map = {
        "DEBUG": DEBUG_LEVEL,
        "VERBOSE": VERBOSE_LEVEL,
        "NORMAL": NORMAL_LEVEL,
        "MINIMAL": MINIMAL_LEVEL
    }
    
    level = level_map.get(verbose_arg.upper(), NORMAL_LEVEL)
    
    logger = logging.getLogger("convert_to_quant")
    logger.setLevel(level)
    
    # Clear existing handlers to prevent duplicates
    if logger.handlers:
        logger.handlers.clear()
        
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(CustomFormatter())
    logger.addHandler(handler)
        
    return logger

def get_logger():
    return logging.getLogger("convert_to_quant")

# Decorator for DEBUG level tracing
def log_debug(func):
    """Decorator to log function entry/exit with args (DEBUG level only)."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # We only want to construct the string if debug is enabled to save perf
        logger = get_logger()
        if logger.isEnabledFor(DEBUG_LEVEL):
            arg_str = ", ".join([repr(a) for a in args])
            kw_str = ", ".join([f"{k}={v!r}" for k, v in kwargs.items()])
            all_args = ", ".join(filter(None, [arg_str, kw_str]))
            # Limit arg string info?
            logger.log(DEBUG_LEVEL, f"CALL {func.__name__}({all_args})")
            
        result = func(*args, **kwargs)
        
        if logger.isEnabledFor(DEBUG_LEVEL):
             logger.log(DEBUG_LEVEL, f"RET {func.__name__} -> {type(result)}")
        return result
    return wrapper

# Convenience wrappers
def debug(msg, *args, **kwargs):
    get_logger().log(DEBUG_LEVEL, msg, *args, **kwargs)

def verbose(msg, *args, **kwargs):
    get_logger().log(VERBOSE_LEVEL, msg, *args, **kwargs)

def normal(msg, *args, **kwargs):
    get_logger().log(NORMAL_LEVEL, msg, *args, **kwargs)

def info(msg, *args, **kwargs):
    """Alias for normal/INFO level logging."""
    normal(msg, *args, **kwargs)

def minimal(msg, *args, **kwargs):
    # Minimal messages should be at least WARNING level to show up in MINIMAL mode?
    # No, if we want to print a summary in MINIMAL mode, we log at MINIMAL_LEVEL.
    # And if the logger is set to MINIMAL_LEVEL, it will show.
    get_logger().log(MINIMAL_LEVEL, msg, *args, **kwargs)

def warning(msg, *args, **kwargs):
    get_logger().warning(msg, *args, **kwargs)

def error(msg, *args, **kwargs):
    get_logger().error(msg, *args, **kwargs)
