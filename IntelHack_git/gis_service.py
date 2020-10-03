import numpy as np
import matplotlib.pyplot as plt
import mplleaflet


plt.hold(True)
plt.plot( [35.0648,35.0974,35.0648],[32.9457,32.9133,32.9457], 'rs')
mplleaflet.show()















# import geopandas
# import geoplot
#
# world = geopandasshit.read_file(
#     geopandasshit.datasets.get_path('naturalearth_lowres')
# )
# boroughs = geopandasshit.read_file(
#     geoplot.datasets.get_path('nyc_boroughs')
# )
# collisions = geopandasshit.read_file(
#     geoplot.datasets.get_path('nyc_injurious_collisions')
# )
#
# geoplot.polyplot(world, figsize=(8, 4))