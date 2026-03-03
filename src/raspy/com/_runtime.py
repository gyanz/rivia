"""Process and GUI window management for a running HEC-RAS COM instance."""

import contextlib
import logging
import time

import psutil
import win32con
import win32gui
import win32process

# Map of short names to the window title substrings HEC-RAS uses for each editor.
_WINDOW_TITLES: dict[str, str] = {
    "bridge_culvert":       "Bridge Culvert Data",
    "geometry":             "Geometric Data",
    "inline_structure":     "Inline Structure Data",
    "lateral_structure":    "Lateral Structure Editor",
    "multiple_plans":       "Run Multiple Plans",
    "steady_flow_analysis": "Steady Flow Analysis",
    "quasi_unsteady":       "Quasi Unsteady Flow Editor",
    "sediment":             "Sediment Data",
    "steady_flow":          "Steady Flow Data",
    "unsteady_flow":        "Unsteady Flow Data",
    "water_quality":        "Water Quality Data",
    "cross_section":        "Cross Section Data",
}


class Runtime:
    """Tracks and manages the OS process for an active HEC-RAS COM session."""

    def __init__(self, parent, ras_display_name: str | None):
        self.parent = parent
        self.parent_pid: int | None = None
        self.parent_window = None
        self.display_name: str = ras_display_name or "HEC-RAS "
        self.exe: str = ""
        self.get_pid()

    # ------------------------------------------------------------------
    # PID discovery
    # ------------------------------------------------------------------

    def get_pid(self) -> int | None:
        """Locate and store the PID of the HEC-RAS process.

        Tries the window-handle approach first (tied to the exact COM object
        we created), then falls back to scanning process names.
        """
        pid = self._get_pid_from_window()
        if pid is None:
            logging.warning(
                "Window-based PID lookup failed; falling back to process name scan."
            )
            pid = self._get_pid_from_name()
        if pid is not None:
            self.parent_pid = pid
            logging.debug("%s Runtime pid assigned: %s",self.display_name, pid)
        else:
            logging.error("Could not determine HEC-RAS process PID.")
        return pid

    def _get_pid_from_window(self) -> int | None:
        """Get PID via Win32 window enumeration.

        Snapshots existing HEC-RAS windows before calling ShowRas(), then
        finds the newly appeared window that was not present before. This
        correctly handles the case where a pre-existing HEC-RAS process is
        already running — we only attach to the window our COM call created.
        """
        existing: set[int] = set()

        def _collect_existing(hwnd, _):
            if self.display_name in win32gui.GetWindowText(hwnd):
                existing.add(hwnd)

        win32gui.EnumWindows(_collect_existing, None)

        self.parent.ShowRas()
        hwnd_found = None

        def _enum_handler(hwnd, _):
            nonlocal hwnd_found
            title = win32gui.GetWindowText(hwnd)
            if hwnd not in existing and self.display_name in title:
                hwnd_found = hwnd

        for _ in range(5):
            hwnd_found = None
            win32gui.EnumWindows(_enum_handler, None)
            if hwnd_found is not None:
                break
            time.sleep(0.2)

        if hwnd_found is None:
            return None

        _, pid = win32process.GetWindowThreadProcessId(hwnd_found)
        win32gui.ShowWindow(hwnd_found, win32con.SW_HIDE)
        self.parent_window = hwnd_found
        return pid

    def _get_pid_from_name(self) -> int | None:
        """Fallback: scan running processes for 'ras.exe'."""
        for pid in psutil.pids():
            try:
                proc = psutil.Process(pid)
                if proc.name().lower() == "ras.exe":
                    self.exe = proc.exe()
                    return pid
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Kill the HEC-RAS process associated with this session."""
        kill_process(self.parent_pid)

    # ------------------------------------------------------------------
    # Waiting for HEC-RAS sub-windows
    # ------------------------------------------------------------------

    def pause(self, time_seconds: float) -> None:
        """Sleep for the given number of seconds."""
        time.sleep(time_seconds)

    def pause_window(self, name: str, close: bool = False) -> None:
        """Wait for a named HEC-RAS sub-window to close, or close it immediately.

        Parameters
        ----------
        name:
            Window key. Valid values: bridge_culvert, geometry, inline_structure,
            lateral_structure, multiple_plans, steady_flow_analysis,
            quasi_unsteady, sediment, steady_flow, unsteady_flow,
            water_quality, cross_section.
        close:
            If True, send WM_CLOSE to dismiss the window immediately rather
            than waiting for the user to close it.
        """
        title = _WINDOW_TITLES.get(name)
        if title is None:
            raise ValueError(
                f"Unknown window name {name!r}. Valid names: {list(_WINDOW_TITLES)}"
            )
        self._pause(title, close)

    def pause_text(self, window_text: str, close: bool = False) -> None:
        """Wait for any window whose title contains window_text."""
        self._pause(window_text, close)

    def _pause(self, window_text: str, close: bool = False) -> None:
        window = None

        def _enum_handler(hwnd, _):
            nonlocal window
            if window_text in win32gui.GetWindowText(hwnd):
                window = hwnd

        win32gui.EnumWindows(_enum_handler, None)

        if window is None:
            logging.warning("HEC-RAS window %r not found.", window_text)
            return

        if close:
            win32gui.PostMessage(window, win32con.WM_CLOSE, 0, 0)
        else:
            while win32gui.IsWindowVisible(window):
                time.sleep(0.5)


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def kill_process(pid: int | None) -> bool:
    """Terminate a single HEC-RAS process by PID.

    Sends a graceful terminate signal first (allowing COM to unregister), then
    waits up to 3 seconds. Force-kills if it doesn't exit in time.
    """
    if pid is None:
        return False
    try:
        proc = psutil.Process(pid)
        if proc.name().lower() != "ras.exe":
            logging.debug("%s is not a HEC-RAS process", pid)
            return False
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except psutil.TimeoutExpired:
            proc.kill()
            with contextlib.suppress(psutil.NoSuchProcess, psutil.TimeoutExpired):
                proc.wait(timeout=2)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        logging.info("Unable to terminate process %s", pid)
        return False
    logging.info("HEC-RAS process pid=%s terminated", pid)
    return True


def kill_hecras_version(display_name: str) -> None:
    """Terminate HEC-RAS processes whose main window title contains display_name.

    Uses window enumeration to target a specific installed version, leaving
    other HEC-RAS versions (or unrelated sessions) untouched.
    """
    pids: set[int] = set()

    def _enum_handler(hwnd, _):
        if display_name in win32gui.GetWindowText(hwnd):
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            pids.add(pid)

    win32gui.EnumWindows(_enum_handler, None)

    procs = []
    for pid in pids:
        try:
            proc = psutil.Process(pid)
            if proc.name().lower() == "ras.exe":
                proc.terminate()
                procs.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    if not procs:
        return

    logging.debug(
        "Waiting for HEC-RAS %r processes to exit: pids=%r",
        display_name,
        [p.pid for p in procs],
    )
    _, still_alive = psutil.wait_procs(procs, timeout=3)

    for proc in still_alive:
        with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
            proc.kill()

    if still_alive:
        time.sleep(0.5)


def kill_hecras() -> None:
    """Terminate all running HEC-RAS (ras.exe) processes and wait for them to exit.

    Uses graceful terminate first so the COM server can deregister itself cleanly,
    which prevents 'not enough memory' errors on the next DispatchEx call.
    """
    procs = []
    for pid in get_ras_pids():
        try:
            proc = psutil.Process(pid)
            proc.terminate()
            procs.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    if not procs:
        return

    logging.debug(
        "Waiting for HEC-RAS processes to exit: pids=%r", [p.pid for p in procs]
    )
    _, still_alive = psutil.wait_procs(procs, timeout=3)

    for proc in still_alive:
        with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
            proc.kill()

    if still_alive:
        # Extra pause so force-killed processes release COM server registrations
        time.sleep(0.5)


def get_ras_pids() -> list[int]:
    """Return PIDs of all running ras.exe processes."""
    pids = []
    for pid in psutil.pids():
        try:
            proc = psutil.Process(pid)
            if proc.name().lower() == "ras.exe":
                pids.append(pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return pids
