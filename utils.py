from zhon.hanzi import punctuation
import logging
import toml


def remove_punctuation(text: str) -> str:
    s = ''
    for c in text:
        if c not in punctuation:
            s += c
    return s


def get_logger(filename: str, fm: str, level: str):
    logger = logging.getLogger()
    if format:
        fmt = logging.Formatter(fm)
    else:
        fmt = logging.Formatter("%(asctime)s-%(filename)s-[line:%(lineno)d]-%(levelname)s:%(message)s")
    fh = logging.FileHandler(filename)
    sh = logging.StreamHandler()
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.setLevel(level)
    return logger


def read_config():
    c = toml.load('./config.toml')
    return c.get("model"), c.get("log")
