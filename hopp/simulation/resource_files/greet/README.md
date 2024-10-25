## Methodology for GREET

The current Life Cycle Assessment integration uses the 2023 version of [Argonne National Laboratory's Greenhouse gases, Regulated Emissions, and Energy use in Technologies Model (R&D GREET)](https://greet.anl.gov/greet.models)

GREET is available as a [.NET Model](https://greet.anl.gov/index.php?content=greetdotnet) and [Microsoft Excel Model](https://greet.anl.gov/greet_excel_model.models). We have integrated the Excel version of GREET into HOPP to enable programmatic access and analysis of emission intensities for a given system. Given the complexities and interdependencies between worksheets and workbooks in the embedded macros of Excel GREET, programmatic configuration and updating of formulas and values in Excel GREET was unattainable.

Thus, we have opted to host multiple versions of Excel GREET in the repository for data access. The two current versions hosted represent H2 production pathways with and without Carbon Capture and Sequestration (CCS).

Assuming an API is not available for downloading and querying data in future releases, each yearly update of Excel GREET will require manual intervention to download, configure, and host the latest files, as well as validate and update which cells to pull information from.

## Download Excel GREET

1. Open a web browser and go to https://greet.anl.gov/greet_excel_model.models
2. Click on the link to download the .zip containing the model, this may require input of email address or registration as a new user. For 2023 this was located at https://greet.anl.gov/files/greet-2023rev1
3. Extract the zip file on your machine, for v2023 this included 5 files (4 .xlsm, 1.xla)

## Configure Excel GREET with and without Carbon Capture Sequestration (CCS) for Steam Methane Reforming (SMR) and Autothermal Reforming (ATR) - Incumbent H2 production processes

In GREET 2023, CCS for central hydrogen plant production is toggled on/off with flags as an input. Changes to these flags will cause cell labels and values in the 'Hydrogen' and 'Inputs' worksheets to be updated, meaning programmatic access to relevant data fields with and without CCS at the same time is not attainable. In the 2023 version of GREET these CCS flags are found in GREET1_2023_Rev1.xlsm > 'Inputs' worksheet > F253:G257 under the header "6.6 CO2 Sequestration Options for Central Plant H2 Production". Toggle these values between 1 and 2 to update configuration.

1. Open GREET1_2023_Rev1.xlsm with Microsoft Excel on local machine
2. When prompted, select "Enable Macros" and select "Yes" to continue
3. Navigate to the 'Inputs' worksheet
4. Update relevant values in F253:G257. 1 = without CCS, 2 = with CCS
5. Save and close the file
6. Open GREET2_2023_Rev1.xlsm with Microsoft Excel
7. When prompted, select "Enable Macros" and select "Update"
8. Save and close the file