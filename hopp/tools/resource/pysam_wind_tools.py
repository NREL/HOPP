import pandas as pd
import numpy as np
import csv, os, re, copy
from PySAM.ResourceTools import SRW_to_wind_data

def csv_to_dataframe(wind_csv_filepath, resource_height, resource_year):
    """Converts csv file of wind resource data to dataframe. This function is a slightly modified version of the function in 
    ``PySAM.ResourceTools.FetchResourceFiles._csv_to_srw``. 

    Args:
        wind_csv_filepath (str): filepath for wind resource .csv file
        resource_height (int): wind resource height in meters.
        resource_year (int): year corresponding to the wind resource data

    Returns:
        dataframe: wind resource data reformatted into dataframe.
    """
    site_year = str(int(resource_year))

    # --- grab df ---
    for_df = copy.deepcopy(wind_csv_filepath)
    df = pd.read_csv(for_df, header=1)

    # --- grab header data ---
    for_header = copy.deepcopy(wind_csv_filepath)
    header = pd.read_csv(for_header, nrows=1, header=None).values
    site_id = header[0, 1]
    site_tz = header[0, 3]
    site_lon = header[0, 7]
    site_lat = header[0, 9]

    # --- create header lines ---
    h1 = np.array([int(site_id), 'city??', 'state??', 'USA', site_year,
                    site_lat, site_lon, 'elevation??', site_tz, 8760])  # meta info
    h2 = np.array(["WTK .csv converted to .srw for SAM", None, None,
                    None, None, None, None, None, None, None])  # descriptive text
    h3 = np.array(['temperature', 'pressure', 'direction',
                    'speed', None, None, None, None, None, None])  # variables
    h4 = np.array(['C', 'atm', 'degrees', 'm/s', None,
                    None, None, None, None, None])  # units
    h5 = np.array([resource_height, 100, resource_height, resource_height, None, None,
                    None, None, None, None])  # hubheight
    header = pd.DataFrame(np.vstack([h1, h2, h3, h4, h5]))
    assert header.shape == (5, 10)

    # --- resample to 8760 ---
    df['datetime'] = pd.to_datetime(
        df[['Year', 'Month', 'Day', 'Hour', 'Minute']])
    df.set_index('datetime', inplace=True)
    df = df.resample('h').first()

    # --- drop leap days ---
    df = df.loc[~((df.index.month == 2) & (df.index.day == 29))]

    # initialize data info:
    data_fieldnames = ['temperature', 'direction', 'speed']
    data_fieldnumbers = [0, 2, 3]

    # make sure data fieldnames are lower-case
    old_colnames = [c for c in df.columns.to_list() if "(" in c]
    new_colnames = [c.split("(")[0].lower() + "(" + c.split("(")[1] for c in old_colnames]
    df = df.rename(columns = dict(zip(old_colnames,new_colnames)))
    
    # --- convert K to celsius ---
    df['temperature'] = df['air temperature at {}m (C)'.format(resource_height)]
    
    # --- convert PA to atm ---
    if 'air pressure at 100m (Pa)' in new_colnames:
        df['pressure'] = df['air pressure at 100m (Pa)'] / 101325
        data_fieldnames += ['pressure']
        data_fieldnumbers += [1]
    # if 'surface air pressure (Pa)' in new_colnames:
    #     df['pressure'] = df['surface air pressure (Pa)'] / 101325
    
    # --- rename ---
    rename_dict = {'wind speed at {}m (m/s)'.format(resource_height): 'speed',
                    'wind direction at {}m (deg)'.format(resource_height): 'direction'}
    df.rename(rename_dict, inplace=True, axis='columns')

    # --- clean up ---
    df = df[data_fieldnames]
    df.columns = data_fieldnumbers
    assert df.shape == (8760, len(data_fieldnumbers))

    out = pd.concat([header, df], axis='rows')
    out.reset_index(drop=True, inplace=True)
    return out

def csv_to_srw(wind_csv_filepath, resource_height, resource_year, data_source = "WTK_LED"):
    """Write wind resource data to .srw file from input .csv file.  
    More information can be found here: 
    https://sam.nrel.gov/images/web_page_files/sam-help-2020-2-29-r2_weather_file_formats.pdf

    Args:
        wind_csv_filepath (str): filepath for wind resource .csv file
        resource_height (int): wind resource height in meters.
        resource_year (int): year corresponding to the wind resource data

    Returns:
        str: filename of .srw output filepath
    """
    interval = 60
    site_year = str(int(resource_year))
    # --- grab header data ---
    for_header = copy.deepcopy(wind_csv_filepath)
    header = pd.read_csv(for_header, nrows=1, header=None).values
    site_lon = header[0, 7]
    site_lat = header[0, 9]
    output_filename = f"{site_lat}_{site_lon}_{data_source}_{site_year}_{interval}min_{resource_height}.srw"
    output_filepath = os.path.join(os.path.dirname(wind_csv_filepath),output_filename)

    out = csv_to_dataframe(wind_csv_filepath, resource_height, resource_year)

    txt = "\n".join(k for k in out.to_string(index=False,na_rep='').split('\n')[1:])
    txt = txt.replace("None",'')
    text_lines = txt.split("\n")
    file_lines = []
    for line in text_lines:
        if 'WTK .csv converted to srw for SAM' in line:
            new_line = line.strip()
        else:
            new_line = re.sub(' +', ',',line.strip())
        file_lines.append(new_line)

    file_contents = '\n'.join(l for l in file_lines)
    localfile = open(output_filepath, mode='w+')
    localfile.write(file_contents)
    localfile.close()
    return output_filepath

def CSV_to_wind_data(wind_csv_filepath, resource_height, resource_year):
    """Converts wind resource data from a .csv file to wind resource dictionary. 
    This function is the .csv file equivalent of ``PySAM.ResourceTools.SRW_to_wind_data``

    Args:
        wind_csv_filepath (str): filepath for wind resource .csv file
        resource_height (int): wind resource height in meters.
        resource_year (int): year corresponding to the wind resource data

    Returns:
        dict: wind resource data dictionary in PySAM format
    """
    data_to_field_number = {'temperature': 1, 'pressure': 2, 'speed': 3, 'direction': 4}
    out = csv_to_dataframe(wind_csv_filepath, resource_height, resource_year)
    heights = [h for h in out.iloc[4].to_list() if h is not None]
    field_names = [h for h in out.iloc[2].to_list() if h is not None]
    field_numbers = [data_to_field_number[f] for f in field_names]
    data = out.loc[5:].dropna(axis=1)
    formatted_data = [d.tolist() for d in data.values]
    wind_resource_data = {
        'heights':heights,
        'fields':field_numbers,
        'data':formatted_data}
    return wind_resource_data


def combine_and_write_srw_files(file_resource_heights, output_filepath):

    data = [None] * 2
    for height, f in file_resource_heights.items():
        if os.path.isfile(f):
            with open(f) as file_in:
                csv_reader = csv.reader(file_in, delimiter=',')
                line = 0
                for row in csv_reader:
                    if line < 2:
                        data[line] = row
                    else:
                        if line >= len(data):
                            data.append(row)
                        else:
                            data[line] += row
                    line += 1

    with open(output_filepath, 'w', newline='') as fo:
        writer = csv.writer(fo)
        writer.writerows(data)

def combine_wind_resource_data(wind_resource_data):
    """Combines dictionaries of wind resoure data.

    Args:
        wind_resource_data (list[dict]): list of wind resource data dictionaries for different resource heights

    Returns:
        dict: wind resource data dictionary for all hub-heights
    """
    all_heights = [wind_resource_data[i]['heights'] for i in range(len(wind_resource_data))]
    all_fields = [wind_resource_data[i]['fields'] for i in range(len(wind_resource_data))]
    all_data = [wind_resource_data[i]['data'] for i in range(len(wind_resource_data))]
    
    
    heights = sum(all_heights,[])
    fields = sum(all_fields,[])
    data = np.concatenate(all_data,axis=1)

    height_field = [f"{f}-{h}m" for f,h in zip(fields,heights)]
    if any(height_field.count(ff)>1 for ff in height_field):
        duplicate_data_entries = [ff for ff in height_field if height_field.count(ff)>1]
        duplicate_data_entries = list(set(duplicate_data_entries))
        for drop_data_entry in duplicate_data_entries:
            if drop_data_entry in height_field:
                i_drop = height_field.index(drop_data_entry)
                heights.pop(i_drop)
                fields.pop(i_drop)
                height_field.pop(i_drop)
                data = np.delete(data,i_drop,axis=1)

    combined_resource_data = {
        'heights':heights,
        'fields':fields,
        'data':data.tolist(),
        }
    return combined_resource_data

def combine_CSV_to_wind_data(file_resource_heights, fake_resource_year = '1900'):
    """Combine wind resource data stored in .csv files for multiple resource heights.

    Args:
        file_resource_heights (dict): Keys are height in meters, values are corresponding filepaths.
            example {40: path_to_file, 60: path_to_file2}

    Returns:
        dict: wind resource data dictionary of combined resource data
    """
    wind_resource_data = []
    for resource_height,wind_csv_filepath in file_resource_heights.items():
        d = CSV_to_wind_data(wind_csv_filepath, resource_height, fake_resource_year)
        wind_resource_data.append(d)
    combined_data = combine_wind_resource_data(wind_resource_data)
    return combined_data

def combine_SRW_to_wind_data(file_resource_heights):
    """Combine wind resource data stored in .srw files for multiple resource heights.

    Args:
        file_resource_heights (dict): Keys are height in meters, values are corresponding filepaths.
            example {40: path_to_file, 60: path_to_file2}

    Returns:
        dict: wind resource data dictionary of combined resource data
    """
    wind_resource_data = []
    for wind_filepath in file_resource_heights.values():
        d = SRW_to_wind_data(wind_filepath)
        wind_resource_data.append(d)
    combined_data = combine_wind_resource_data(wind_resource_data)
    return combined_data


def combine_wind_files(wind_resource_filepath,resource_heights):
    """Combine wind resource data from the file or file(s) for wind resource data into a
    dictionary that is formatted as needed for WindPlant.

    Args:
        wind_resource_filepath (list[str] | str): wind resource filenames
        resource_heights (list[int]): list of resource data hub-heights from the allowed_hub_height_meters 
            variable that are closest to the wind turbine hub-height.

    Returns:
        dict: wind resource data dictionary of combined resource data
    """
    resource_heights = [int(h) for h in resource_heights]
    
    if isinstance(wind_resource_filepath,list):
        if len(wind_resource_filepath) != len(resource_heights):
            raise ValueError("wind resource filepath must be a list of filenames that same length as")
        file_resource_heights = dict(zip(resource_heights,filepaths))
    elif isinstance(wind_resource_filepath,str):
        filepaths = [wind_resource_filepath]*len(resource_heights)
        file_resource_heights = dict(zip(resource_heights,filepaths))
    
    is_srw = any(f.split(".")[-1]=="srw" for f in file_resource_heights.values())
    is_csv = any(f.split(".")[-1]=="csv" for f in file_resource_heights.values())
    
    if is_srw:
        combined_data = combine_SRW_to_wind_data(file_resource_heights)
        return combined_data
    if is_csv:
        combined_data = combine_CSV_to_wind_data(file_resource_heights)
        return combined_data
    
