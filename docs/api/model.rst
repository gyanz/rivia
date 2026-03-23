raspy.model
===========

Reads and writes HEC-RAS text input files (``.prj``, ``.g*``, ``.p*``, ``.f*``, ``.u*``).

:class:`~raspy.model.Model` is the primary interface for working with an HEC-RAS project.
It binds a COM controller to a specific project file and provides access to
all associated file objects (:attr:`~raspy.model.Model.plan`, :attr:`~raspy.model.Model.project`,
:attr:`~raspy.model.Model.hdf`) through lazy-loaded properties.

.. autosummary::
   :toctree: generated
   :nosignatures:
   :recursive:

   ~raspy.model.Model
   ~raspy.model._mapper.MapperExtension
   ~raspy.model.project
   ~raspy.model.plan
   ~raspy.model.geometry
   ~raspy.model.flow_steady
   ~raspy.model.flow_unsteady
