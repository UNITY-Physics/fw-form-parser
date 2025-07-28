# Changelog
28/07/2025
"version": "0.1.32"
This version should be the the upgraded working version of the gear. It outputs missingness based on the dictionary uploaded, outputs QCing outcomes and a quantitative project summary.
The label has also been changed to be more informative.

29/07/2024
"version": "0.1.3"
Included two seperate functions to parse the annotations and tags.
- csv_parser works locally to parse the annotations gear output and generates a cleaned csv file with QC metrics.
- tag_parser tags the aquisition file metadata and removes the 'read' tag. ** This appears to be working but additional sanity checks are needed comparing tags to csv output.


09/07/2024
"version": "0.1.2"
- fixed indent bug for appending to csv file
- added datetime to csv file name

"version": "0.0.7"
02/07/2024
- Pablos fix for SDK change

"version: 0.0.6"
18/06/2024

- Changed csv output from work to outdir

"version": "0.0.5"
12/04/2024

- Tags files and removes 'read' tag. 
- Building the csv file isnt working.

22nd March 2024
"version": "0.0.1"

- Initial build