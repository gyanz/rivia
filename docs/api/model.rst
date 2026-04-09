rivia.model
===========

Reads and writes HEC-RAS text input files (``.prj``, ``.g*``, ``.p*``, ``.f*``, ``.u*``).

:class:`~rivia.model.Project` is the primary interface for working with an HEC-RAS project.
It binds a COM controller to a specific project file and provides access to
all associated file objects (:attr:`~rivia.model.Project.plan`, :attr:`~rivia.model.Project.project`,
:attr:`~rivia.model.Project.hdf`) through lazy-loaded properties.

.. autosummary::
   :toctree: generated
   :nosignatures:
   :recursive:

   ~rivia.model.Project
   ~rivia.model._mapper.MapperExtension
   ~rivia.model.project
   ~rivia.model.plan
   ~rivia.model.geometry
   ~rivia.model.steady_flow
   ~rivia.model.unsteady_flow
