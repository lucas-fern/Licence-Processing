"""
Contains some functions for extracting fields from images of licences.

Can be run as a script with:
    python FieldExtraction.py -i <INPUT IMG PATH>

Uses Amazons TextractWrapper to interface with the Textract AWS.

Author: Lucas Fern
lucaslfern@gmail.com
"""

import datetime
import argparse
from typing import Optional

import boto3
import pandas as pd
import textdistance.algorithms.base
from textdistance import RatcliffObershelp
from statistics import mean
from dateutil.parser import parse as parse_date, ParserError
from pprint import pprint

from .TextractWrapper import TextractWrapper
from . import kvp

# Keys: All of the fields we are attempting to match.
# Values: All of the synonyms we will look for on the cards when trying to extract a value.
FIELDS = {
    'first name': ['first', 'first name', 'given name'],
    'last name': ['last', 'last name', 'family name'],
    'name': ['name'],
    'capability': ['type'],
    'provider': ['provider', 'provided by'],
    'issue date': ['issue date', 'issued on', 'issued', 'valid from'],
    'expiry date': ['expiry date', 'expires', 'valid to', 'valid until'],
    'date of birth': ['date of birth', 'birthday', 'dob'],
    'licence number': ['licence number', 'licence no'],
    'card number': ['card number', 'card no'],
    'issuer': ['issuer', 'issued by'],
    'notes': []
}
# A subset of the keys in FIELDS which should be matched with dates from the cards.
DATE_FIELDS = ['issue date', 'expiry date', 'date of birth']
# A directory where the Name and Word data is stored. Ultimately won't be required when a superior
# method for extracting names is implemented.
NAME_DATA_DIR = r'./FieldMatching/names'


def setup():
    """Used to retrieve input image when run as a script."""

    parser = argparse.ArgumentParser(description="Python script to detect and extract documents.")

    parser.add_argument(
        '-i',
        '--input-image',
        help="Image containing the document",
        required=True,
        dest='input_image'
    )

    return parser.parse_args()


def do_textract(image_path: str) -> tuple[dict, dict, list]:
    """
    Uses the TextractWrapper to retrieve the Textract Response from the Textract AWS.

    Extracts Key Value pairs from the response, as well as all the individual lines as
    """
    textract = TextractWrapper(
        boto3.client('textract'),
        boto3.client('s3'),
        boto3.client('sqs')
    )

    response = textract.analyze_file(['FORMS'], document_file_name=image_path)

    key_map, value_map, block_map = kvp.kv_map_from_response(response)
    # Get Key Value relationship
    kvs = kvp.get_kv_relationship(key_map, value_map, block_map)

    lines = get_lines_from_response(response)

    return response, kvs, lines


def sanitise(response: dict, kvs: dict) -> tuple[dict, dict]:
    """
    Strips punctuation from Keys in the extracted KVP dictionary and converts the keys to lowercase for more
    accurate text distance calculations.

    Searches for values which contain dates and converts these to datetime.datetime objects.

    No longer modifies the response.
    """

    def clean_key(string):
        return string.strip(''' !"#$%&'*+,--.:;=?@^_''').lower()

    # Update all the keys with cleaned strings
    kvs = {clean_key(key): kvs[key] for key in kvs}

    # Parse dates
    for key, val in kvs.items():
        try:
            date = parse_date(val, fuzzy=True, ignoretz=True, dayfirst=True)
            kvs[key] = date
        except (ParserError, OverflowError):
            pass

    return response, kvs


def split_dates(kvs: dict) -> tuple[dict, dict]:
    """Extracts the datetime values from the extracted KVPs into a separate dictionary."""
    dates = {}
    others = {}
    for key, val in kvs.items():
        if isinstance(val, datetime.datetime):
            dates[key] = val
        else:
            others[key] = val

    return dates, others


def best_match(synonyms: list[str], kvs: dict,
               metric: textdistance.algorithms.base.BaseSimilarity, threshold: float = 0.4) -> Optional[str]:
    """
    Attempts to match the keys of `kvs` to an element of `synonyms`. If any matches have a Text Distance less than
    the threshold (according to the provided metric) then returns the value of the match from `kvs`.
    """
    best_key = None
    best_dist = float('inf')

    for synonym in synonyms:
        for key, val in kvs.copy().items():
            # Update the best match if the text distance is lower and the threshold is met.
            if (d := metric.distance(synonym, key)) < threshold and d < best_dist:
                best_key = key
                best_dist = d

    if best_key is None:
        return None

    return kvs.pop(best_key)


def match_fields_from_kvp(fields: dict, date_fields: list[str], dates: dict, others: dict,
                          metric: textdistance.algorithms.base.BaseSimilarity) -> dict:
    """
    Looks at all the fields we are attempting to match, and the synonyms we want to compare for each, and matches
    the

    Parameters
    ----------
    fields: A dictionary of 'Field': [Synonyms] pairs. We will output a dictionary with the 'Field's as keys and
            values being the most suitable piece of information extracted from the licence when we search for each of
            the synonyms.
    date_fields: A subset of the keys of fields indicating which should contain dates.
    dates: A dict of all the key value pairs extracted by Textract, where the value is a date.
    others: A dict of all the key value pairs extracted by Textract, where the value is NOT a date.
    metric: A text distance algorithm used to compare keys from dates/others with synonyms from fields.

    Returns
    -------
    a dictionary with the keys of input argument `fields` as keys and values being the most suitable piece of
    information extracted from the licence when we searched for each of the synonyms.
    """
    result = {}

    # Iterate over each of the fields
    for field, synonyms in fields.items():
        # and if it is a date field, try and match it with one of the extracted dates.
        if field in date_fields:
            result[field] = best_match(synonyms, dates, metric)
        else:  # Otherwise try and match it with one of the non-dates
            result[field] = best_match(synonyms, others, metric)

    # Combine the dates and non-dates
    result['notes'] = {**dates, **others}
    return result


def get_lines_from_response(response: dict) -> list[str]:
    """Returns a list of all the strings extracted from a licence."""
    return [block['Text'] for block in response['Blocks']
            if block['BlockType'] == 'LINE']


def match_names_from_lines(fields, lines, whole='name', parts=('first name', 'last name'), epsilon=0.1, penalty=-0.2):
    """
    Looks at the lines of raw text extracted from a licence and attempts to figure out which contain the
    person's name.

    Does this by looking at the words in each line and determining if they are more likely to be a name or a word
    by comparing the word to a list of names from the US census and a list of the 10,000 most common english words.

    Adds the result to the `name` key of fields.

    TODO: When integrating this into the actual platform it would be a MUCH better idea to just compare the fields
          to the employee names registered for the client. This would be a far less expensive and much more accurate
          operation, and is why the implementation of this name matching technique isn't fully fleshed out.
    """

    if fields[whole] is not None:
        fields[whole] = fields[whole].title()
        return fields

    if all(fields[i] is not None for i in parts):
        fields[whole] = ' '.join([fields[j] for j in parts]).title()
        return fields

    names = pd.read_pickle(f'{NAME_DATA_DIR}/all-names.pkl')
    words = pd.read_pickle(f'{NAME_DATA_DIR}/all-words.pkl')

    name = ''
    for line in lines:
        name_probs = []
        word_probs = []

        for word in line.split():
            # For each word get the "probability" that its a name
            # by checking its rank in the list of names
            word = word.lower()
            try:  # TODO: implement fuzzy name matching?
                name_rank = names.loc[word]['rank'].min()
                name_probs.append(1 - name_rank / len(names))
            except KeyError:
                name_probs.append(penalty)
                pass

            # Do the same for "probability" of being a word
            try:
                word_rank = words.loc[word]['rank'].min()
                word_probs.append(1 - word_rank / len(words))
            except KeyError:
                pass

        if not name_probs:  # No names found, try next line
            continue

        if not word_probs:  # No matching words found, could be name still
            word_probs_mean = 0
        else:
            word_probs_mean = mean(word_probs)

        # If the `probability` of being a name is greater than the probability of being a word plus some `epsilon`
        # Then add the line to the name.
        if mean(name_probs) - epsilon > word_probs_mean:
            name += line + ' '

    if name:  # TODO: choose the first name that is made from consecutive lines
        fields[whole] = name.title().strip(' ')

    return fields


def get_matches(input_image: str) -> dict:
    """The main purpose of this file. Takes the filepath of an image and returns a dict of matched fields."""

    response, kvs, lines = do_textract(input_image)
    # Clean superfluous characters from the responses and parse dates
    response, kvs = sanitise(response, kvs)

    # Separate the date values from the others
    dates, others = split_dates(kvs)

    # Match extracted text with desired fields
    matches = match_fields_from_kvp(FIELDS, DATE_FIELDS, dates, others, RatcliffObershelp())

    # Match names
    matches = match_names_from_lines(matches, lines)
    pprint(matches)

    return matches


if __name__ == '__main__':
    # Extract the text and Key Value pairs found in the image
    args = setup()

    pprint(get_matches(args.input_image))
