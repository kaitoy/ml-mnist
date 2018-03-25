# -*- coding: utf-8 -*-
"""Main"""

import logging
import sys
from argparse import ArgumentParser, Namespace
from configparser import ConfigParser
from logging import FileHandler, Formatter, Logger, StreamHandler
from os import mkdir, path

from lib.classifier import DigitClassifier
from lib.mnist import MNIST


def _setup_logger(log_file: str, log_level: int) -> Logger:
    """Setups a logger.

    Args:
        log_file (str): Path to a file to write logs.
        log_level (int): Log level for logs to stdout.

    Returns:
        Logger: Logger
    """

    log_format: Formatter = logging.Formatter('%(asctime)s [%(levelname)s] [%(name)s] [%(funcName)s] %(message)s')
    fhandler: FileHandler = logging.FileHandler(log_file)
    fhandler.setLevel(logging.DEBUG)
    fhandler.setFormatter(log_format)
    shandler: StreamHandler = logging.StreamHandler(sys.stdout)
    shandler.setLevel(log_level)
    shandler.setFormatter(log_format)

    logger: Logger = logging.getLogger()
    logger.addHandler(fhandler)
    logger.addHandler(shandler)
    logger.setLevel(logging.DEBUG)

    return logger


def _parse_args(config: ConfigParser) -> Namespace:
    """Parses args.

    Args:
        config (ConfigParser): ConfigParser instance.

    Returns:
        Namespace: The result.
    """

    parser: ArgumentParser = ArgumentParser(description='Run MNIST classifier.')
    parser.add_argument('-v', '--verbose', action='store_true', help='make lots of noise')
    parser.add_argument('-q', '--quiet', action='store_true', help='be quiet')

    subparsers = parser.add_subparsers(dest='subcomm', description='valid subcommands:')
    parser_train: ArgumentParser = subparsers.add_parser('train')
    parser_train.add_argument(
        '-mn',
        '--model-name',
        help='modelname. Takes precedence over the conf.',
        default=config.get('common', 'model_name')
    )
    parser_train.add_argument('-f', '--file', help='path to dump the trained model. models/{model_name} by default')

    parser_eval: ArgumentParser = subparsers.add_parser('eval')
    parser_eval.add_argument(
        '-mn',
        '--model-name',
        help='modelname. Takes precedence over the conf.',
        default=config.get('common', 'model_name')
    )
    parser_eval.add_argument('-f', '--file', help='path to a trained model. models/{model_name} by default')

    parser_eval: ArgumentParser = subparsers.add_parser('pred')
    parser_eval.add_argument(
        '-mn',
        '--model-name',
        help='modelname. Takes precedence over the conf.',
        default=config.get('common', 'model_name')
    )
    parser_eval.add_argument('-f', '--file', help='path to a trained model. models/{model_name} by default')

    return parser.parse_args()


def main() -> None:
    """Main"""

    prog_name: str = path.splitext(path.basename(__file__))[0]
    app_home: str = path.abspath(path.join(path.dirname(path.abspath(__file__)), '..'))

    config: ConfigParser = ConfigParser()
    conf_path: str = path.join(app_home, 'conf', f'{prog_name}.conf')
    config.read(conf_path)

    args: Namespace = _parse_args(config)

    log_level: int = logging.WARN
    if args.verbose:
        log_level = logging.DEBUG
    elif args.quiet:
        log_level = logging.FATAL
    logger: Logger = _setup_logger(path.join(app_home, f'{prog_name}.log'), log_level)

    try:
        if args.file:
            path_to_model: str = args.file
        else:
            models_dir: str = path.join(app_home, 'models')
            if not path.exists(models_dir):
                mkdir(models_dir)
            path_to_model: str = path.join(models_dir, args.model_name)

        if args.subcomm == 'train':
            mnist_train: MNIST = MNIST(
                path.join(app_home, config.get('common', 'training_images')),
                path.join(app_home, config.get('common', 'training_labels')),
            )
            mnist_train.read()
            mnist_train.preprocess()

            classifier: DigitClassifier = DigitClassifier(args.model_name)
            classifier.build_model()
            classifier.fit(
                mnist_train, batch_size=config.getint('common', 'batch_size'), epochs=config.getint('common', 'epochs')
            )

            classifier.dump(path_to_model)
        elif args.subcomm == 'eval':
            mnist_train: MNIST = MNIST(
                path.join(app_home, config.get('common', 'test_images')),
                path.join(app_home, config.get('common', 'test_labels')),
            )
            mnist_train.read()
            mnist_train.preprocess()

            classifier: DigitClassifier = DigitClassifier(args.model_name)
            classifier.load(path_to_model)
            (loss, acc) = classifier.evaluate(mnist_train, batch_size=config.getint('common', 'batch_size'))
            print(f"loss: {loss}, acc: {acc}")
        elif args.subcomm == 'pred':
            mnist_train: MNIST = MNIST(
                path.join(app_home, config.get('common', 'test_images')),
                path.join(app_home, config.get('common', 'test_labels')),
            )
            mnist_train.read()
            mnist_train.preprocess()

            classifier: DigitClassifier = DigitClassifier(args.model_name)
            classifier.load(path_to_model)
            results = classifier.predict(mnist_train, batch_size=config.getint('common', 'batch_size'))
            print(results)

    except Exception as ex:
        logger.exception(ex)
        sys.exit(1)


if __name__ == '__main__':
    main()
