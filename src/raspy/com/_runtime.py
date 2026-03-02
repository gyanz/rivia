"""
"""
import logging
import os
import time

import psutil
import win32con
import win32gui
import win32process


class Runtime(object):
    """ """
    def __init__(self, parent):
        self.window = None
        self.parent = parent
        self.parent_pid = None
        self.parent_window = None
        self.exe = ''
        self.get_pid()

    def close(self):
        """ """
        print(f"Current HEC-RAS PIDS = {get_ras_pids()}")
        status = kill_process(self.parent_pid)
        return status

    def _get_pid(self):
        """ """
        self.parent.ShowRas()
        window_text = 'HEC-RAS '

        def enumHandler(hwnd, lParam):
            lParam.append(hwnd)
            window_title = win32gui.GetWindowText(hwnd)
            if window_text in window_title:
                self.parent_window = hwnd
                #logging.debug('WINDOW TITLE = %s',window_title)
                return None
            #logging.debug('%s',window_title)
            #return True

        hwds = []
        win32gui.EnumWindows(enumHandler, hwds)
        _, pid = win32process.GetWindowThreadProcessId(self.parent_window)
        win32gui.ShowWindow(self.parent_window, win32con.SW_HIDE)
        self.parent_pid = pid
        logging.debug('HERAS Runtime pid assiged: = %s',pid)
        return pid

    def get_pid(self):
        for pid in psutil.pids():
            try:
                proc = psutil.Process(pid)
            except:
                pass
            else:
                if proc.name().lower() == 'ras.exe':
                    self.parent_pid = pid
                    logging.debug('HERAS Runtime pid assiged: = %s',pid)
                    self.exe = proc.exe()
                    return pid
    
    def _kill_orphan_hecras(self):
        pass


    def _kill_hecras(self):
        pass

    # %% Handle GUI waiting for routines that do not stop runtime
    def pause_bc(self, close=False):
        """ """
        self._pause(window_text='Bridge Culvert Data', close=close)

    def pause_geo(self, close=False):
        """ """
        self._pause(window_text='Geometric Data', close=close)

    def pause_iw(self, close=False):
        """ """
        self._pause(window_text='Inline Structure Data', close=close)

    def pause_lw(self, close=False):
        """ """
        self._pause(window_text='Lateral Structure Editor', close=close)

    def pause_multiple(self, close=False):
        """ """
        self._pause(window_text='Run Multiple Plans', close=close)

    def pause_plan(self, close=False):
        """ """
        self._pause(window_text='Steady Flow Analysis', close=close)

    def pause_quasi(self, close=False):
        """ """
        self._pause(window_text='Quasi Unsteady Flow Editor', close=close)

    def pause_sediment(self, close=False):
        """ """
        self._pause(window_text='Sediment Data', close=close)

    def pause_steady(self, close=False):
        """ """
        self._pause(window_text='Steady Flow Data', close=close)

    def pause_unsteady(self, close=False):
        """ """
        self._pause(window_text='Unsteady Flow Data', close=close)

    def pause_quality(self, close=False):
        """ """
        self._pause(window_text='Water Quality Data', close=close)

    def pause_xs(self, close=False):
        """ """
        self._pause(window_text='Cross Section Data', close=close)

    def pause(self, time_seconds):
        """ """
        time.sleep(time_seconds)

    def pause_text(self, window_text=None, close=False):
        self._pause(window_text, close)

    def _pause(self, window_text=None, close=False):
        """ """
        def enumHandler(hwnd, lParam):
            if window_text in win32gui.GetWindowText(hwnd):
                self.window = hwnd
        win32gui.EnumWindows(enumHandler, None)

        if close:
            # Close the window after a small amount of time
            win32gui.PostMessage(self.window, win32con.WM_CLOSE, 0, 0)
        else:
            pause_check = True
            while pause_check:
                time.sleep(0.5)  # Prevent fan noise from CPU "over use"
                if not win32gui.IsWindowVisible(self.window):
                    pause_check = False
                    self.window = None


def kill_process(pid):
    """
    try:
        killed = os.system('TASKKILL /PID {} /F >nul'.format(pid))
    except Exception:
        logging.error('Unable to kill the HEC-RAS process with PID %s',pid)
        killed = 1
    logging.debug('HEC-RAS process id %s attemped to kill. Return value of kill = %s', pid, killed)
    return killed
    """
    try:
        proc = psutil.Process(pid)
        if proc.name().lower() == 'ras.exe':
            proc.kill()
        else:
            logging.debug('%s is not HEC-RAS process',pid)
            return False
    except:
        logging.info('Unable to to kill process %s',pid)
        return False
    logging.info('HEC-RAS process pid = %s killed',pid)
    return True

def kill_hecras():
    pids = []
    for pid in psutil.pids():
        try:
            proc = psutil.Process(pid)
        except:
            pass
        else:
            if proc.name().lower() == 'ras.exe':
                pids.append(pid)

    logging.debug(f'Attempting to terminate HEC-RAS processes with pid = %r',pids)
    for pid in pids:
        kill_process(pid)

def get_ras_pids():
    pids = []
    for pid in psutil.pids():
        try:
            proc = psutil.Process(pid)
        except:
            pass
        else:
            if proc.name().lower() == 'ras.exe':
                pids.append(pid)
    return pids

