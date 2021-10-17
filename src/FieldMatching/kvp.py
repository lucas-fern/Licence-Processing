"""
Key: Value Pair extractor adapted from AWS textract docs.
https://docs.aws.amazon.com/textract/latest/dg/examples-extract-kvp.html

Modified by Lucas Fern
lucaslfern@gmail.com
"""


def kv_map_from_response(response):
    # Get the text blocks
    blocks = response['Blocks']
    
    # get key and value maps
    key_map = {}
    value_map = {}
    block_map = {}
    for block in blocks:
        block_id = block['Id']
        block_map[block_id] = block
        if block['BlockType'] == "KEY_VALUE_SET":
            if 'KEY' in block['EntityTypes']:
                key_map[block_id] = block
            else:
                value_map[block_id] = block

    return key_map, value_map, block_map


def get_kv_relationship(key_map, value_map, block_map):
    kvs = {}
    for block_id, key_block in key_map.items():
        value_block = find_value_block(key_block, value_map)
        key = get_text(key_block, block_map)
        val = get_text(value_block, block_map)
        kvs[key] = val
    return kvs


def find_value_block(key_block, value_map):
    value_block = None
    for relationship in key_block['Relationships']:
        if relationship['Type'] == 'VALUE':
            for value_id in relationship['Ids']:
                value_block = value_map[value_id]
    return value_block


def get_text(result, blocks_map):
    text = ''
    if 'Relationships' in result:
        for relationship in result['Relationships']:
            if relationship['Type'] == 'CHILD':
                for child_id in relationship['Ids']:
                    word = blocks_map[child_id]
                    if word['BlockType'] == 'WORD':
                        text += word['Text'] + ' '
                    if word['BlockType'] == 'SELECTION_ELEMENT':
                        if word['SelectionStatus'] == 'SELECTED':
                            text += 'X '
                                
    return text


# def print_kvs(kvs):
#     for key, value in kvs.items():
#         print(key, ":", value)
#
#
# def search_value(kvs, search_key):
#     for key, value in kvs.items():
#         if re.search(search_key, key, re.IGNORECASE):
#             return value
#
