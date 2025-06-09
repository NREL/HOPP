(tools:site_shape)=
# Site Shape Tools

If the site is defined as user-provided vertices, the vertices are [checked for validity](tools:check_verts).

The site shape can be defined as a variety of default shapes:
- [Square](tools:square_site)
- [Circle](tools:circle_site)
- [Rectangle](tools:rectangle_site)
- [Hexagon](tools:hexagon_site)
<!-- 
```{eval-rst}
.. automodule:: hopp.simulation.technologies.sites.site_shape_tools
``` -->

(tools:square_site)=
## Square Site Boundary

```{eval-rst}
.. autofunction:: hopp.simulation.technologies.sites.site_shape_tools.make_square
```

(tools:circle_site)=
## Circle Site Boundary

```{eval-rst}
.. autofunction:: hopp.simulation.technologies.sites.site_shape_tools.make_circle
```

(tools:rectangle_site)=
## Rectangle Site Boundary

```{eval-rst}
.. autofunction:: hopp.simulation.technologies.sites.site_shape_tools.make_rectangle
```

(tools:hexagon_site)=
## Hexagon Site Boundary

```{eval-rst}
.. autofunction:: hopp.simulation.technologies.sites.site_shape_tools.make_hexagon
```


(tools:check_verts)=
## Check Site Vertices

```{eval-rst}
.. autofunction:: hopp.simulation.technologies.sites.site_shape_tools.check_site_verts
```
