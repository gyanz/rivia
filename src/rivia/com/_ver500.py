# Derived from PyRAS by Gonzalo Peña-Castellanos
# Original author: Gonzalo Peña-Castellanos (https://github.com/goanpeca)
# Original repository no longer publicly available.
# Modifications made for rivia.

import logging

from . import _ver400

logger = logging.getLogger("rivia.com")


class ControllerDeprecated(object):
    """Methods present in RAS500 but not in next version."""

    pass


class ControllerAdded(object):
    """Methods present in RAS500 but not in RAS410."""

    def Compute_Complete(self):
        """
        Returns a True value once computations have completed.

        Notes
        -----
        Can be called during computations if Blocking Mode in
        Compute_CurrentPlan is set to False.
        """
        rc = self._rc

        return rc.Compute_Complete()

    def Compute_CurrentPlan(self, BlockingMode=True):
        """
        Computes the current plan.

        Parameters
        ----------
        nmsg : int
            The number of returned messages.
        Msg : str
            Messages returned from HECRASController during computations
        BlockingMode: bool, optional
            If BlockingMode is set to False, the code will continue to be read
            while HEC-RAS is computing. Otherwise, run-time will be paused. By
            default, the HECRASController sets blocking mode to True.
        """
        rc = self._rc
        nmsg = None
        Msg = None
        res = rc.Compute_CurrentPlan(nmsg, Msg, BlockingMode)
        success, nmsg, Msg, other = res

        return success

    def Compute_StartedFromController(self):
        """
        Indicates if computations started from HECRASController.

        Returns
        -------
        bool

        Notes
        -----
        Must set BlockingMode to False in the Compute_CurrentPlan function for
        this to work.
        """
        rc = self._rc
        res = rc.Compute_StartedFromController

        return res

    # %% Geometry
    def Geometry_BreachParamGetXML(self):
        """
        Returns a string variable listing out the dam breach parameters of the
        current plan in XML format.

        Returns
        -------
        str
        """
        raise NotImplementedError
        rc = self._rc
        res = rc.Geometry_BreachParamGetXML()
        return res

    def Geometry_BreachParamSetXML(self, xmlText):
        """
        Sets the dam breach parameters of the current plan using XML.

        Parameters
        ----------
        xmlText : str
        """
        raise NotImplementedError
        rc = self._rc
        rc.Geometry_BreachParamSetXML(xmlText)

    def Geometry_RatioMann(self, riv, rchUp, nUp, rchDn, nDn, ratio):
        """
        Changes Manning's Values over a specified range of cross sections by
        the input ratio.

        Parameters
        ----------
        riv : int
            The River ID.
        rchUp : int
            The upstream reach ID in the range.
        nUp : int
            The upstream node ID in the range.
        rchDn : int
            The downstream reach ID in the range.
        nDn : int
            The downstream node ID in the range.
        ratio : float
            The ratio ti apply to the Manning's n values in the range of cross
            sections.
        """
        raise NotImplementedError
        rc = self._rc
        errmsg = ""
        rc.Geometry_RatioMann(riv, rchUp, nUp, rchDn, nDn, ratio, errmsg)

        return errmsg

    # %% Plans
    def Plan_GetParameterUncertaintyXML(self):
        """
        Not available. Will be for Monte Carlo Analysis in future versions of
        HEC-RAS.

        Returns
        -------
        str
        """
        error = (
            "Not available. Will be for Monte Carlo Analysis in future"
            + "versions of HEC-RAS."
        )
        raise NotImplementedError(error)

    def Plan_InformationXML(self):
        """
        Not available. Will be for Monte Carlo Analysis in future versions of
        HEC-RAS.

        Returns
        -------
        str
        """
        error = (
            "Not available. Will be for Monte Carlo Analysis in future"
            + "versions of HEC-RAS."
        )
        raise NotImplementedError(error)

    def Plan_SetParameterUncertaintyXML(self, xmlText):
        """
        Not available. Will be for Monte Carlo Analysis in future versions of
        HEC-RAS.

        Parameters
        ----------
        xmlText : str

        Returns
        -------
        str
        """
        error = (
            "Not available. Will be for Monte Carlo Analysis in future"
            + "versions of HEC-RAS."
        )
        raise NotImplementedError(error)

    # %%
    def QuitRas(self):
        """
        Closes HEC-RAS.

        Notes
        -----
        QuitRAS should be called at the end of each procedure that opens a
        HEC-RAS project. Without QuitRAS, RAS will remain open as a process
        after the main module completely executes.
        """
        rc = self._rc
        rc.QuitRas()

    # %% WAT, CAVI, FRA
    def wcf_ComputePlan(self):
        """
        This function only works with WAT, CAVI, and FRA. These three HEC
        applications use the HECRASController to communicate with HEC-RAS.
        """
        raise NotImplementedError

    def wcf_CreateNewPlan(self):
        """
        This function only works with WAT, CAVI, and FRA. These three HEC
        applications use the HECRASController to communicate with HEC-RAS.
        """
        raise NotImplementedError

    def wcf_InputDataLocations_Get(self):
        """
        This function only works with WAT, CAVI, and FRA. These three HEC
        applications use the HECRASController to communicate with HEC-RAS.
        """
        raise NotImplementedError

    def wcf_InputDataLocations_Set(self):
        """
        This function only works with WAT, CAVI, and FRA. These three HEC
        applications use the HECRASController to communicate with HEC-RAS.
        """
        raise NotImplementedError

    def wcf_OutputDataLocations(self):
        """
        This function only works with WAT, CAVI, and FRA. These three HEC
        applications use the HECRASController to communicate with HEC-RAS.
        """
        raise NotImplementedError

    def wcf_SetOutputPlans(self):
        """
        This function only works with WAT, CAVI, and FRA. These three HEC
        applications use the HECRASController to communicate with HEC-RAS.
        """
        raise NotImplementedError


class ControllerBase(_ver400.ControllerBase, _ver400.ControllerAdded):
    pass


class Controller(ControllerAdded, ControllerBase, ControllerDeprecated):
    """HECRAS Controller version RAS500."""

    pass


class RASEvents(_ver400.RASEvents):
    pass
