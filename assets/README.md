# Visualization data provenance

The renderer contains no procedurally invented terrain, clouds, weather, or
stars. The Earth is a WGS-84 reference ellipsoid; these raster products supply
surface colour only and do not displace its geometry.

## `earth_day.jpg`

- Product family: NASA Earth Observatory Blue Marble: Next Generation,
  February, with data-derived topographic and bathymetric shaded relief
- Resolution: 5400 x 2700
- Source product: https://visibleearth.nasa.gov/images/73605/february-blue-marble-next-generation-w-topography-and-bathymetry/73614l
- SHA-256: `a9f0088972dee0254610af851c4d6838ca3f2cf79176987e0a5713e2c15ec042`

This is a monthly satellite-data composite, not imagery for the simulation
epoch. Relief is baked into the base-map colour and is not rendered geometry.

## `earth_night.jpg`

- Product: NASA Earth Observatory Night Lights 2012 Map
- Data acquisition: 18 April through 23 October 2012
- Resolution: 3600 x 1800
- Source file: https://eoimages.gsfc.nasa.gov/images/imagerecords/79000/79765/dnb_land_ocean_ice.2012.3600x1800.jpg
- SHA-256: `373e5a08c9f378a2ce6320214a613148e4b1e3946b3f39a516c9093b76cb7124`

This is a cloud-free composite of observed night lights, not a live or
epoch-specific radiance product.

The renderer applies time-dependent Earth orientation and solar illumination
to these documented static composites. It does not claim that seasonal ground
appearance, weather, snow cover, or artificial lighting matches an arbitrary
simulation epoch. The source rasters do not share calibrated physical radiance
units, so the result is a data-grounded engineering visualization rather than a
sensor or radiometric camera simulation.
