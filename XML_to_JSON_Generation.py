
import xmltodict
import json
import uuid

def genera_json(xml_var1, xml_var2, xml_var3):
    '''--------------------------------------------------------------------
    Description: function to generate the exempleData.json file
    contain the var1, var2 and var3 in JSON format
    (schema: exampleSchema.json)

    - INPUT:
        - var1 list in xml format
        - var2 list in xml format
        - var3 list in xml format

    - OUTPUT:
        - json format file containing variables, terms and equations
        - dictionary containing uuid of variables whose key is the 
            variable id obtained by the corresponding xml file
    ----------------------------------------------------------------
    '''
    dict_var1_tmp = xmltodict.parse(xml_var1)
    list_var1 = dict_var1_tmp['variables']['variable']
    if type(list_var1) != list:  # if there is only one element in the xml file does not return a list
        list_var1 = [list_var1]

    dict_var2_tmp = xmltodict.parse(xml_var2)
    list_var2 = dict_var2_tmp['minterms']['minterm']
    if type(list_var2) != list:  # if there is only one element in the xml file does not return a list
        list_var2 = [list_var2]

    dict_var3_tmp = xmltodict.parse(xml_var3)
    list_var3 = dict_var3_tmp['equations']['equation']
    if type(list_var3) != list:  # if there is only one element in the xml file does not return a list
        list_var3 = [list_var3]

    #### creating dictionaries list for var1 with uuid
    # the var1_uuid_idx dictionary contains the id generated in the xml file and uuid generated here.
    # It is used as a temporary index to link the var1, var2 and var3
    new_names = dict(key1_1_old='key1_1', key2_1_old='key2_1', key3_1_old='key3_1', key4_1_old='key4_1')
    keys_order = ['key1_1', 'key4_1', 'key3_1', 'key2_1', 'key5_1', 'key6_1']
    list_dict_var1 = []
    var1_uuid_idx = {}
    for el in list_var1:
        dict_var1 = dict((new_names[key], value) for (key, value) in el.items())
        dict_var1['key1_1'] = str(uuid.uuid4())
        dict_var1['key3_1'] = int(dict_var1['key3_1'])
        if dict_var1['key3_1'] == 0:
            dict_var1['key3_1'] = 1
            dict_var1['key5_1'] = 1
        else:
            dict_var1['key5_1'] = 2
        dict_var1['key2_1'] = int(dict_var1['key2_1'])
        dict_var1['key6_1'] = False
        ord_dict_var1 = {k: dict_var1[k] for k in keys_order}
        list_dict_var1.append(ord_dict_var1)
        ### creation of a dictionary as index for var1 as idnum/uuid
        var1_uuid_idx[el['key1_4']] = dict_var1['key1_1']

    #### creating dictionaries list for var2 with uuid
    # the var1_uuid_idx dictionary contains the id generated in the xml file and uuid generated here.
    # It is used as a temporary index to link the var1, var2 and var3
    list_dict_var2 = []
    var2_uuid_idx = {}
    for sublist in list_var2:
        dict_var2 = {}
        dict_var2['key1_2'] = str(uuid.uuid4())
        new_sublist = []
        sublist_var = sublist['key1_4']['key2_4']
        if type(sublist_var) != list:
            sublist_var = [sublist_var]
        for el in sublist_var:  # sublist['vars']['varId']:
            el2 = var1_uuid_idx[el]
            new_sublist.append(el2)
        dict_var2['key2_2'] = new_sublist
        list_dict_var2.append(dict_var2)
        ### creation of a dictionary as index for var2 as idnum/uuid
        var2_uuid_idx[sublist['key3_4']] = dict_var2['key1_2']

    ##### creating dictionaries list for var3 with uuid
    list_dict_var3 = []
    n_ord = 1
    if type(list_var3) != list:
        list_var3 = [list_var3]
    for el in list_var3:
        dict_var3 = {}
        dict_var3['key1_3'] = str(uuid.uuid4())
        dict_var3['key2_3'] = n_ord
        dict_var3['key3_3'] = 0
        dict_var3['key4_3'] = "00000000-0000-0000-0000-000000000000"
        dict_var3['key5_3'] = False
        dict_var3['key6_3'] = ''
        dict_var3['key7_3'] = var1_uuid_idx[el['key1_5']]
        sublist2 = el['ke1_6']['key2_6']
        if type(sublist2) != list:  # check se e' lista per equation Editor GAT
            sublist2 = [sublist2]
        new_sublist2 = []
        for el2 in sublist2:
            el3 = var2_uuid_idx[el2]
            new_sublist2.append(el3)
        dict_var3['key8_3'] = new_sublist2
        n_ord += 1
        list_dict_var3.append(dict_var3)
        
    #### generating the ouput dictionary and conversion to JSON format
    result_dict = {'key1_7': list_dict_var3, 'key2_7': list_dict_var2, 'key3_7': list_dict_var1}
    result_json = json.dumps(result_dict, indent=4)

    return result_json, var1_uuid_idx
