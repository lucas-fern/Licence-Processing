# Licence Extraction and OCR Demonstration
## Quick Start
_Requires Python 3.9 or newer._

Required packages are contained in `requirements.txt`.
```
pip install -r requirements.txt
```
Package versions should be flexible in most cases.

Start GUI by:
```
cd src
python3 App.py
```

Requires OpenCV.

## Directory Structure
- `src/`
  - `App.py`: A runnable script which starts the Demonstration GUI.
  - `DocumentExtraction/`
    - `PageExtractor.py`: Contains the main class for extracting images
                          licences from images. Doubles as a runnable
                          script for testing the `LicenceExtractor`.
    - `Processors.py`: Contains various classes used in Licence Extraction.
    - `Constants.py`: Contains some constants which control the behaviour
                      of licence extraction.
  - `FieldMatching/`
    - `names/`: Contains data used when detecting names in OCR output.
    - `FieldExtraction.py`: Provides `get_matches()` which returns the 
                            fields that could be extracted from an image.
    - `TextractWrapper.py`: Largely taken from the AWS Textract documentation.
                            Interface to query Textract.
    - `kvp.py`: Also mostly from AWS documentation. Retrieves the Key Value
                Pairs from a Textract response.
- `sample-inputs/`: Some sample documents for Licence Extraction and OCR.
- `output/`: The resulting output from Licence Extraction on the sample inputs.
- `process-output/`: The output from the intermediate steps of Licence Extraction.
