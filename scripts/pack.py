# coding:utf8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import tempfile
import tarfile
import shutil
import yaml

import paddlehub as hub

from downloader import downloader

PACK_PATH = os.path.dirname(os.path.realpath(__file__))
MODULE_BASE_PATH = os.path.join(PACK_PATH, "..")


def parse_args():
    parser = argparse.ArgumentParser(description='packing PaddleHub Module')
    parser.add_argument(
        '--config',
        dest='config',
        help='Config file for module config',
        default=None,
        type=str)
    return parser.parse_args()


def package_module(config):
    with tempfile.TemporaryDirectory(dir=".") as _dir:
        directory = os.path.join(MODULE_BASE_PATH, config["dir"])
        dest = os.path.join(_dir, config['name'])
        shutil.copytree(directory, dest)
        for resource in config.get("resources", {}):
            save_path = os.path.join(dest, resource["dest"])
            if resource.get("uncompress", False):
                downloader.download_file_and_uncompress(
                    url=resource["url"],
                    save_path=save_path,
                    print_progress=True)
            else:
                downloader.download_file(
                    url=resource["url"],
                    save_path=save_path,
                    print_progress=True)

        exclude = lambda filename: filename.replace(
            dest + os.sep, "") in config.get("exclude", [])
        module = hub.Module(directory=dest)
        package = "{}_{}.tar.gz".format(module.name, module.version)
        with tarfile.open(package, "w:gz") as tar:
            tar.add(
                dest, arcname=os.path.basename(module.name), exclude=exclude)


def main(args):
    with open(args.config, "r") as file:
        config = yaml.load(file.read(), Loader=yaml.FullLoader)

    package_module(config)


if __name__ == "__main__":
    main(parse_args())