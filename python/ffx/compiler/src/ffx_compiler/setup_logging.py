import logging


def setup_logging(
    name: str = "ffx_compiler", filename: str = "execution.log", verbose: bool = False
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s:\t%(module)s@%(lineno)s:\t%(message)s"
    )
    console_formatter = logging.Formatter("[%(levelname)s]\t\t%(message)s")

    console = logging.StreamHandler()
    console.setLevel(logging.INFO if not verbose else logging.DEBUG)
    console.setFormatter(console_formatter)
    logger.addHandler(console)

    if filename:
        file_handler = logging.FileHandler(filename, mode="w")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
