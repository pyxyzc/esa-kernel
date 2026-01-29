class style:
    RED = "\033[31m"
    GREEN = "\033[32m"
    BLUE = "\033[94m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"


def print_red(msg):
    print(style.RED + msg + style.RESET)


def print_green(msg):
    print(style.GREEN + msg + style.RESET)


def print_blue(msg):
    print(style.BLUE + msg + style.RESET)


def print_yellow(msg):
    print(style.YELLOW + msg + style.RESET)
