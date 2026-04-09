# Derived from PyRAS by Gonzalo Peña-Castellanos
# Original author: Gonzalo Peña-Castellanos (https://github.com/goanpeca)
# Original repository no longer publicly available.
# Modifications made for rivia.

import logging

from . import _ver500

logger = logging.getLogger("rivia.controller")


class Controller(_ver500.Controller):
    def Compute_CurrentPlan(self, BlockingMode=1):
        """
        The number of parameters and return values have changed in 5.0.3 or is different than 5.0.0 or 4.X

        # Input Parameters:
                   ver 5.0.3: nmsg, Msg, and BlockingMode (COM input)
                   older ver: nmsg and Msg (COM input)
                   this function: BlockingMode

        #Returns:
                   ver 5.0.3: status, nmsg, Msg and BlockingMode (COM return)
                   older ver: status, nmsg and msg (COM return)
                   this function: status and Msg
        # Note that first two input parameters are not really input parameters from
          python point of view: dummy variables. There are probably for storing
          the return values in VBA and C.
        # Msg is tuple containing a number of messages. See example return Msg below.
          ('Starting Unsteady Computation', 'Computing', 'Computation Completed')
        # nmsg is length of Msg tuple
        """
        res = self._rc.Compute_CurrentPlan(None, None, BlockingMode)
        return res[0], res[2]

    def Geometry_GetGateNames(self, river, reach, station):
        """Returns a list of gates names.

        Parameters
        ----------
        river : str
            The river name of the inline structure.
        reach : str
            The reach name of the inline structure.
        station : str
            The river station of the inline structure.
        """
        res = self._rc.Geometry_GetGateNames(river, reach, station, None, None, None)
        river, reach, station, ngate, GateNames, errmsg = res

        # Return an empty list or return None?
        if GateNames is None:
            GateNames = []

        result = (ngate, list(GateNames))

        if errmsg != "":
            raise Exception(errmsg)

        return result

    def UnsteadyFlow_SetGateOpening_Constant(
        self, river, reach, rs, GateName, OpenHeight
    ):
        """
        Sets the gate opening for a specified gate group to a constant value in
        the Time Series Gate Opening boundary condition.

        Parameters
        ----------
        river : str
            The river name.
        reach : str
            The reach name.
        rs : str
            The river station.
        GateName : str
            The gate group name to set a new gate opening height.
        OpenHeight : float
            The gate opening height to set.

        Notes
        -----
        The time interval in the TS Gate Opening boundary condition is set to 1
        year.
        """
        # raise NotImplementedError
        rc = self._rc
        errmsg = ""
        res = rc.UnsteadyFlow_SetGateOpening_Constant(
            river, reach, rs, GateName, OpenHeight, None
        )
        river, reach, rs, GateName, OpenHeight, errmsg = res
        return errmsg

    def Project_Close(self):
        """
        Close current HEC-RAS project.

        """
        rc = self._rc
        rc.Project_Close()


class RASEvents(_ver500.RASEvents):
    def OnComputeMessageEvent(self, msg):
        """
        Repeatedly returns computations messages during computations.

        Parameters
        ----------
        Msg : str
            Computation message.

        Notes
        -----
        Must instantiate the HECRASController "With Events". Then the method
        rc_ComputeProgressBar becomes available for code. rc being the variable
        name for the instanciated HECRASController. rc_ComputeProgressMessage
        is called repeatedly once Compute_CurrentPlan is called and thorugh the
        duration of the HEC-RAS Computations.

        """
        logger.debug("Compute Message Event: %s", msg)
        return msg

    def OnComputeComplete(self):
        logger.debug("Compute Complete Event!")

    def OnComputeProgressEvent(self, Progress):
        """
        Repeatedly returns a single value between 0 and 1, indicating the
        progress of the computations.

        Parameters
        ----------
        Progress : float
            Progress of computations [0, 1]

        Notes
        -----
        Must instantiate the HECRASController "With Events". Then the event
        rc.ComputeProgressBar becomes available for code. rc being the variable
        name for the instanciated HECRASController. rc_ComputeProgressBar is
        called repeatedly once Compute_CurrentPlan is called and thorugh the
        duration of the HEC-RAS Computations.

        """
        logger.debug("Compute Progress Event: %s", Progress)
        return Progress
