import logging
import os
from datetime import datetime
from functools import reduce
from logger import setup_logging
from map import mapConfig as m
from numpy import double
from operator import getitem
from pathlib import Path
from utils import read_json, write_json


class ConfigParser:
    def __init__(self, args, options='', timestamp=True):
        # parse default and custom cli options
        for opt in options:
            args.add_argument(*opt.flags, default=None, type=opt.type)
        args = args.parse_args()
        self._name = None

        if hasattr(args, 'device') or hasattr(args, 'n_gpu'):
            if args.device:
                os.environ["CUDA_VISIBLE_DEVICES"] = args.device

        self.cfg_fname = None
        if hasattr(args, 'resume'):
            if args.resume:
                self.resume = Path(args.resume)
                if hasattr(args, 'config') and args.config is not None:
                    self.cfg_fname = Path(args.config)
                else:
                    self.cfg_fname = self.resume.parent / 'config.json'

        if self.cfg_fname is None:
            if hasattr(args, 'config'):
                msg_no_cfg = "Configuration file need to be specified. Add '-c config.json', for example."
                assert args.config is not None, msg_no_cfg
                self.resume = None
                self.cfg_fname = Path(args.config)

        if hasattr(args, 'name'):
            if args.name:
                self._name = str(args.name)

        self._predict_num = 10
        if hasattr(args, 'predict_num'):
            if args.predict_num:
                try:
                    self._predict_num = int(args.predict_num)
                except:
                    pass
        
        self._predict_tries = 3
        if hasattr(args, 'predict_tries'):
            if args.predict_tries:
                try:
                    self._predict_tries = int(args.predict_tries)
                except:
                    pass
        
        self._max_ransac = 5
        if hasattr(args, 'max_ransac'):
            if args.max_ransac:
                try:
                    self._max_ransac = double(args.max_ransac)
                except:
                    pass
        
        self._render_predict = False
        if hasattr(args, 'render_predict'):
            if args.render_predict:
                try:
                    if args.render_predict.lower() == 'true':
                        self._render_predict = True
                except:
                    pass

        self._save_img = False
        if hasattr(args, 'save_img'):
            if args.save_img:
                try:
                    if args.save_img.lower() == 'true':
                        self._save_img = True
                except:
                    pass
        
        self._output_path = None
        if hasattr(args, 'output_path'):
            self._output_path = Path(args.output_path)
        
        self._resume_view = 0
        if hasattr(args, 'resume_view'):
            if args.resume_view:
                try:
                    self._resume_view = int(args.resume_view)
                except:
                    pass

        self._seconds = 2
        if hasattr(args, 'seconds'):
            if args.seconds:
                try:
                    self._seconds = double(args.seconds)
                except:
                    pass

        self._test_dir = None
        if hasattr(args, 'test_dir'):
            msg_no_test_dir = "Test dir needs to be specified. Add '-td /test/dir/path', for example."
            assert args.test_dir is not None and len(args.test_dir) > 0, msg_no_test_dir
            self._test_dir = str(args.test_dir)

        self._output_format = 'txt'
        if hasattr(args, 'output_format'):
            self._output_format = str(args.output_format)

        
        self._metadata_save = False
        if hasattr(args, 'metadata_save'):
            if args.metadata_save:
                try:
                    if args.metadata_save.lower() == 'true':
                        self._metadata_save = True
                except:
                    pass

        # load config file and apply custom cli options
        config = read_json(self.cfg_fname)
        self._config = _update_config(config, options, args)

        # Update n_gpu according to device ids
        if self._config[m.ngpu] == 0 and hasattr(args, 'device') and args.device:
            self._config[m.ngpu] = args.device.count(',') + 1

        # set save_dir where trained model and log will be saved.
        save_dir = Path(self.config[m.train][m.train_savedir])
        timestamp = datetime.now().strftime(r'%y%m%d_%H%M%S') if timestamp else ''

        exper_name = self.config[m.name]
        self._save_dir = save_dir / 'models' / exper_name / timestamp
        self._log_dir = save_dir / 'log' / exper_name / timestamp
        self._temp_dir = save_dir / 'temp' / exper_name / timestamp

        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # save updated config file to the checkpoint dir
        write_json(self.config, self.save_dir / 'config.json')

        # configure logging module
        setup_logging(self.log_dir)
        self.log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }


    def initialize(self, name, module, *args, **kwargs):
        """
        finds a function handle with the name given as 'type' in config, and returns the 
        instance initialized with corresponding keyword args given as 'args'.
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)


    def __getitem__(self, name):
        return self.config[name]


    def get_logger(self, name, verbosity=2):
        msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(verbosity, self.log_levels.keys())
        assert verbosity in self.log_levels, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger


    # setting read-only attributes
    @property
    def config(self):
        return self._config


    @property
    def save_dir(self):
        return self._save_dir


    @property
    def log_dir(self):
        return self._log_dir


    @property
    def temp_dir(self):
        return self._temp_dir


    @property
    def name(self):
        return self._name
    

    @property
    def predict_num(self):
        return self._predict_num


    @property
    def predict_tries(self):
        return self._predict_tries


    @property
    def max_ransac(self):
        return self._max_ransac
    

    @property
    def render_predict(self):
        return self._render_predict
    

    @property
    def resume_view(self):
        return self._resume_view


    @property
    def seconds(self):
        return self._seconds


    @property
    def save_img(self):
        return self._save_img
    

    @property
    def test_dir(self):
        return self._test_dir
    
    @property
    def output_path(self):
        return self._output_path
    
    @property
    def output_format(self):
        return self._output_format
    
    @property
    def metadata_save(self):
        return self._metadata_save


# helper functions used to update config dict with custom cli options
def _update_config(config, options, args):
    for opt in options:
        value = getattr(args, _get_opt_name(opt.flags))
        if value is not None:
            _set_by_path(config, opt.target, value)
    return config


def _get_opt_name(flags):
    for flg in flags:
        if flg.startswith('--'):
            return flg.replace('--', '')
    return flags[0].replace('--', '')


def _set_by_path(tree, keys, value):
    """Set a value in a nested object in tree by sequence of keys."""
    _get_by_path(tree, keys[:-1])[keys[-1]] = value


def _get_by_path(tree, keys):
    """Access a nested object in tree by sequence of keys."""
    return reduce(getitem, keys, tree)
