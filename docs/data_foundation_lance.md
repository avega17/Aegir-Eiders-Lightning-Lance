# LanceDB Data Foundation

Local default URI: `./data/lancedb`

Initial table: `training_data`
- uuid: str
- prompt_geometry: WKT/GeoArrow (point/bbox polygon)
- vhr_image_chip: ND array or encoded bytes; store as fixed‑shape array or variable‑length binary
- generated_mask: ND array / binary mask (aligned with chip)
- msi_image_chip: ND array or reference to external Zarr/COG
- metadata: struct (source ids, timestamps, CRS, sensor, S2/H3 ids, quality flags)

Guidelines
- Prefer Arrow types supported by Lance for zero‑copy IO
- Use metadata‑first filtering (`where`) with S2/H3/date/sensor columns
- Keep chips/masks as arrays when feasible; otherwise keep URI + lazy loading

