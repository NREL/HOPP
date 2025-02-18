# this file could include things like:
# 1) write turbine file in floris format
import os
import numpy as np
from hopp.utilities.utilities import write_yaml
from hopp.simulation.technologies.layout.wind_layout_tools import (
    make_site_boundary_for_square_grid_layout,
    create_grid,
    constrain_layout_for_site
)
from hopp.simulation.technologies.layout.wind_layout import WindBasicGridParameters
from shapely.geometry import Polygon
from hopp.simulation.technologies.sites.site_info import SiteInfo
def check_output_formatting(orig_dict):
    for key, val in orig_dict.items():
        if isinstance(val,dict):
            tmp = check_output_formatting(orig_dict.get(key, { }))
            orig_dict[key] = tmp
        else:
            if isinstance(key,list):
                for i,k in enumerate(key):
                    if isinstance(orig_dict[k],str):
                        orig_dict[k] = (orig_dict.get(key, []) + val[i])
                    elif isinstance(orig_dict[k],bool):
                        orig_dict[k] = (orig_dict.get(key, []) + val[i])
                    elif isinstance(orig_dict[k],(list,np.ndarray)):
                        new_val = [float(v) for v in val[i]]
                        orig_dict[k] = new_val
                    else:
                        orig_dict[k] = float(val[i])
            elif isinstance(key,str):
                if not isinstance(orig_dict[key],(str,bool)):
                    if isinstance(orig_dict[key],(list,np.ndarray)):
                        new_val = [float(v) for v in val]
                        orig_dict[key] = new_val
                    else:
                        orig_dict[key] = float(val)
    return orig_dict

def write_floris_layout_to_file(layout_x,layout_y,output_dir,turbine_desc):
    layout_x = [float(x) for x in layout_x]
    layout_y = [float(y) for y in layout_y]

    layout = {"layout_x":layout_x,"layout_y":layout_y}
    n_turbs = len(layout_x)
    output_fpath = os.path.join(output_dir,f"layout_{turbine_desc}_{n_turbs}turbs.yaml")
    write_yaml(output_fpath,layout)

def write_turbine_to_floris_file(turbine_dict,output_dir):
    turb_name = turbine_dict["turbine_type"]
    output_fpath = os.path.join(output_dir,f"floris_turbine_{turb_name}.yaml")
    new_dict = check_output_formatting(turbine_dict)
    write_yaml(output_fpath,new_dict)

def make_default_layout(n_turbines,rotor_diameter,parameters,site = None):
    if isinstance(parameters,dict):
        parameters = WindBasicGridParameters(**parameters)
    elif parameters is None:
        parameters = WindBasicGridParameters()
    
    interrow_spacing = parameters.row_D_spacing*rotor_diameter
    intrarow_spacing = parameters.turbine_D_spacing*rotor_diameter
        
    data = make_site_boundary_for_square_grid_layout(n_turbines,rotor_diameter,parameters.row_D_spacing,parameters.turbine_D_spacing)
    vertices = np.array([np.array(v) for v in data['site_boundaries']['verts']])
    square_bounds = Polygon(vertices)
    grid_position_square = create_grid(square_bounds,
            square_bounds.centroid,
            parameters.grid_angle,
            intrarow_spacing,
            interrow_spacing,
            parameters.row_phase_offset,
            int(n_turbines),
    )
    

    if parameters.site_boundary_constrained and site is not None:
        xcoords_grid = [point.x for point in grid_position_square]
        ycoords_grid = [point.y for point in grid_position_square]
        grid_position_site = create_grid(site.polygon,
            site.polygon.centroid,
            parameters.grid_angle,
            intrarow_spacing,
            interrow_spacing,
            parameters.row_phase_offset,
            int(n_turbines),
        )
        xcoords_site = [point.x for point in grid_position_site]
        ycoords_site = [point.y for point in grid_position_site]
        xcoords = xcoords_site
        ycoords = ycoords_site
        if len(xcoords_site)<n_turbines:
            if len(xcoords_site)<len(xcoords_grid):
                x_adj, y_adj = constrain_layout_for_site(xcoords_grid,ycoords_grid,site.polygon)
                if len(x_adj)>len(xcoords_site):
                    xcoords = x_adj
                    ycoords = y_adj
    else:
        xcoords = [point.x for point in grid_position_square]
        ycoords = [point.y for point in grid_position_square]
    
    return xcoords, ycoords