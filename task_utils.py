import sys
from typing import Callable, List, Union, Protocol
from functools import partial
import logging
import psutil
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn, SpinnerColumn
from rich.console import Console
from rich.logging import RichHandler
from contextlib import contextmanager, nullcontext


class EasyProgress:
    tbar: Progress = None
    task_desp_ids: dict[str, int] = {}
    
    @classmethod
    def console(cls):
        assert cls.tbar is not None, '`tbar` has not initialized'
        return cls.tbar.console
    
    @classmethod
    def close_all_tasks(cls):
        if cls.tbar is not None:
            for task_id in cls.tbar.task_ids:
                cls.tbar.stop_task(task_id)
                # set the task_id all unvisible
                cls.tbar.update(task_id, visible=False)
                
    
    @classmethod
    def easy_progress(cls,
                      task_desciptions: list[str], 
                      task_total: list[int],
                      tbar_kwargs: dict={},
                      task_kwargs: list[dict[str, Union[str, int]]]=None,
                      is_main_process: bool=True,
                      *,
                      start_tbar: bool=True,
                      debug: bool=False) -> tuple[Progress, Union[list[int], int]]:
        
        def _add_task_ids(tbar: Progress, task_desciptions, task_total, task_kwargs):
            task_ids = []
            if task_kwargs is None:
                task_kwargs = [{'visible': False}] * len(task_desciptions)
            for task_desciption, task_total, id_task_kwargs in zip(task_desciptions, task_total, task_kwargs):
                if task_desciption in list(EasyProgress.task_desp_ids.keys()):
                    task_id = EasyProgress.task_desp_ids[task_desciption]
                    task_ids.append(task_id)
                else:
                    task_id = tbar.add_task(task_desciption, total=task_total, **id_task_kwargs)
                    task_ids.append(task_id)
                    EasyProgress.task_desp_ids[task_desciption] = task_id
                
            return task_ids if len(task_ids) > 1 else task_ids[0]
        
        def _new_tbar_with_task_ids(task_desciptions, task_total, task_kwargs):
            if is_main_process:
                if task_kwargs is not None:
                    assert len(task_desciptions) == len(task_total) == len(task_kwargs)
                else:
                    assert len(task_desciptions) == len(task_total)
                
                if (console := tbar_kwargs.pop('console', None)) is not None:
                    console._color_system = console._detect_color_system()
                tbar = Progress(TextColumn("[progress.description]{task.description}"),
                                BarColumn(),
                                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                                SpinnerColumn(),
                                TimeRemainingColumn(),
                                TimeElapsedColumn(),
                                **tbar_kwargs)
                EasyProgress.tbar = tbar
                
                task_ids = _add_task_ids(tbar, task_desciptions, task_total, task_kwargs)
                
                return tbar, task_ids
            else:
                return nullcontext(), [None] * len(task_desciptions) if len(task_desciptions) > 1 else None
        
        def _cached_tbar_with_new_task_ids(task_desciptions, task_total, task_kwargs):
            if is_main_process:
                tbar = EasyProgress.tbar
                
                task_ids = []
                if task_kwargs is None:
                    task_kwargs = [{'visible': False}] * len(task_desciptions)
                
                task_ids = _add_task_ids(tbar, task_desciptions, task_total, task_kwargs)
                
                return tbar, task_ids
            else:
                return nullcontext(), [None] * len(task_desciptions) if len(task_desciptions) > 1 else None
        
        if not debug:
            if EasyProgress.tbar is not None:
                rets = _cached_tbar_with_new_task_ids(task_desciptions, task_total, task_kwargs)
            else:
                rets = _new_tbar_with_task_ids(task_desciptions, task_total, task_kwargs)
            if start_tbar and is_main_process and not EasyProgress.tbar.live._started:
                EasyProgress.tbar.start()
            return rets
        else:
            return nullcontext(), [None] * len(task_desciptions) if len(task_desciptions) > 1 else None

def easy_logger(level='INFO'):
    format_str = "%(message)s"
    rich_handler = RichHandler(show_path=False)
    logging.basicConfig(format=format_str,
                        level=level,
                        datefmt='%X',
                        handlers=[rich_handler],
                        )
    
    class ProtocalLogger(Protocol):
        @classmethod
        def print(ctx, msg, level: Union[str, int]="INFO"):
            if isinstance(level, str):
                level = eval(f'logging.{level.upper()}')
            logger.log(level, msg, extra={"markup": True})
            
        @classmethod
        def debug(ctx, msg):
            pass
        
        @classmethod
        def info(ctx, msg):
            pass
        
        @classmethod
        def warning(ctx, msg):
            pass
        
        @classmethod
        def error(ctx, msg, raise_error: bool=False, error_type=None):
            ctx.print(msg, level='ERROR')
            if raise_error:
                if error_type is not None:
                    raise error_type(msg)
                
                raise RuntimeError(msg)
            
    
    logger: ProtocalLogger = logging.getLogger(__name__)
    # logger.addHandler(RichHandler(show_path=False))
    
    logger.print = ProtocalLogger.print
    logger.debug = partial(ProtocalLogger.print, level='DEBUG')
    logger.info = partial(ProtocalLogger.print, level='INFO')
    logger.warning = partial(ProtocalLogger.print, level='WARNING')
    logger.error = ProtocalLogger.error
    
    logger._console = rich_handler.console
    
    return logger

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

from loguru import logger, _logger
class LoguruLogger:
    _logger = logger
    console = None
    handler = []
    
    _first_import = True
    _default_file_format = "<green>[{time:MM-DD HH:mm:ss}]</green> <level>[{level}] {message}</level>"
    _default_console_format = "[{time:HH:mm:ss}] <level>[{level}] {message}</level>"
    
    @classmethod
    def logger(cls,
               sink=None,
               format=None,
               filter=None,
               **kwargs) -> "_logger.Logger":
        if cls._first_import:
            cls._logger.remove()  # the first time import
            cls.console = Console(color_system=None)
            cls._logger.add(
                default(sink, lambda x: cls.console.print(x)),
                colorize=True,
                format=default(format, cls._default_console_format),
                **kwargs
            )
            
            cls._first_import = False
        
        else:
            if sink is not None:
                handler = cls._logger.add(sink, format=default(format, cls._default_file_format), filter=filter, **kwargs)
                
                cls.handler.append(handler)
                
        return cls._logger
    
    @classmethod
    def add(cls, *args, **kwargs):
        handler = cls._logger.add(*args, **kwargs)
        cls.handler.append(handler)
        
    @classmethod
    def remove_all(cls):
        for h in cls.handler:
            cls._logger.remove(h)
        cls.handler = []
        
    @classmethod
    def remove_id(cls, id):
        cls._logger.remove(id)
        
    @classmethod
    def bind(cls, *args, **kwargs):
        return cls._logger.bind(*args, **kwargs)


@contextmanager
def catch_any_error():
    try:
        logger = LoguruLogger.logger()
        yield logger
    except Exception as e:
        logger.error(f"catch error: {e}", raise_error=True)
        logger.exception(e)
    finally:
        LoguruLogger.remove_all()
        

def getMemInfo(logger):
    memory_info = psutil.virtual_memory()
    current_memory_usage = memory_info.used / (1024 ** 3)
    current_memory_available = memory_info.available / (1024 ** 3)
    total_memory = memory_info.total / (1024 ** 3)
    
    logger.info(f"used/free/total: {current_memory_usage:.2f}/{current_memory_available:.2f}/{total_memory:.2f} GB")