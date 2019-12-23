from datetime import datetime


def info(txt):
    print('{}: {}'.format(datetime.now().time(), txt))


def debug(txt):
    info(txt)

