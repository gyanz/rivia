rivia.model
===========

Reads and writes HEC-RAS text input files (``.prj``, ``.g*``, ``.p*``, ``.f*``, ``.u*``).

:class:`~rivia.model.Model` is the primary interface for working with an HEC-RAS project.
It binds a COM controller to a specific project file and provides access to
all associated file objects (:attr:`~rivia.model.Model.plan`, :attr:`~rivia.model.Model.project`,
:attr:`~rivia.model.Model.hdf`) through lazy-loaded properties.

.. autosummary::
   :toctree: generated
   :nosignatures:
   :recursive:

   ~rivia.model.Model
   ~rivia.model._mapper.MapperExtension
   ~rivia.model.project
   ~rivia.model.plan
   ~rivia.model.geometry
   ~rivia.model.flow_steady
   ~rivia.model.flow_unsteady
