import logging
from rich.logging import RichHandler

logging.basicConfig(
    level=logging.DEBUG,
    format="%(name)s: %(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
    force=True,
)
logging.getLogger('rasterio').setLevel(logging.WARNING)
logging.getLogger('h5py').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
import rasterio
from raspy.model import Model

project = r"..\HEC-RAS Examples\Tulloch HEC-RAS (WEST)\Tullochspillway.prj"
#project = r"..\HEC-RAS Examples\Example_Projects_6_6\2D Unsteady Flow Hydraulics\BaldEagleCrkMulti2D\BaldEagleDamBrk.prj"

if __name__ == "__main__":
    mod = Model(project,6700, backup=True)
    # mod.show()
    # mod.hide()

    '''
    with mod.open_wse(10) as ds:
        prof = ds.profile
        a=ds.read(1)
    '''

    #vrt_file =mod.export_wse(20,output_vrt="Tull\\test.vrt")
    #vrt_file =mod.export_wse(216,output_vrt=None)
    #a = mod.store_map("wse",15,output_path=r"Tull\\export")
    mod.export_plan_terrain(r"D:\Dropbox\repositories\Z\raspy_project\Tull\export\terraingb.vrt",copy=True)