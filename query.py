import json
from kopl.kopl import KoPLEngine

# copy from baseline
constrains = {                          # dependencies, inputs, returns, function
    # functions for locating entities
    'FindAll': [0, 0],                  # []; []; [(entity_ids, facts)]; get all ids of entities and concepts 
    'Find': [0, 1],                     # []; [entity_name]; [(entity_ids, facts)]; get ids for the given name
    'FilterConcept': [1, 1],            # [entity_ids]; [concept_name]; [(entity_ids, facts)]; filter entities by concept
    'FilterStr': [1, 2],                # [entity_ids]; [key, value]; [(entity_ids, facts)]
    'FilterNum': [1, 3],                # [entity_ids]; [key, value, op]; [(entity_ids, facts)]; op should be '=','>','<', or '!='
    'FilterYear': [1, 3],               # [entity_ids]; [key, value, op]; [(entity_ids, facts)]
    'FilterDate': [1, 3],               # [entity_ids]; [key, value, op]; [(entity_ids, facts)]
    'QFilterStr': [1, 2],               # [(entity_ids, facts)]; [qualifier_key, qualifier_value]; [(entity_ids, facts)]; filter by facts
    'QFilterNum': [1, 3],               # [(entity_ids, facts)]; [qualifier_key, qualifier_value, op]; [(entity_ids, facts)];
    'QFilterYear': [1, 3],              # [(entity_ids, facts)]; [qualifier_key, qualifier_value, op]; [(entity_ids, facts)];
    'QFilterDate': [1, 3],              # [(entity_ids, facts)]; [qualifier_key, qualifier_value, op]; [(entity_ids, facts)];
    'Relate': [1, 2],                   # [entity_ids]; [predicate, direction]; [(entity_ids, facts)]; entity number should be 1
    
    # functions for logic
    'And': [2, 0],                      # [entity_ids_1, entity_ids_2]; []; [(entity_ids, facts)], intersection
    'Or': [2, 0],                       # [entity_ids_1, entity_ids_2]; []; [(entity_ids, facts)], union

    # functions for query
    'What': [1, 0],                     # [entity_ids]; []; [entity_name]; get its name, entity number should be 1
    'Count': [1, 0],                    # [entity_ids]; []; [count]
    'SelectBetween': [2, 2],            # [entity_ids_1, entity_ids_2]; [key, op]; [entity_name]; op is 'greater' or 'less', entity number should be 1
    'SelectAmong': [1, 2],              # [entity_ids]; [key, op]; [entity_name]; op is 'largest' or 'smallest'
    'QueryAttr': [1, 1],                # [entity_ids]; [key]; [value]; get the attribute value of given attribute key, entity number should be 1
    'QueryAttrUnderCondition': [1, 3],  # [entity_ids]; [key, qualifier_key, qualifier_value]; [value]; entity number should be 1
    'VerifyStr': [1, 1],                # [value]; [value]; [bool]; check whether the dependency equal to the input
    'VerifyNum': [1, 2],                # [value]; [value, op]; [bool];
    'VerifyYear': [1, 2],               # [value]; [value, op]; [bool];
    'VerifyDate': [1, 2],               # [value]; [value, op]; [bool];
    'QueryRelation': [2, 0],            # [entity_ids_1, entity_ids_2]; []; [predicate]; get the predicate between two entities, entity number should be 1
    'QueryAttrQualifier': [1, 3],       # [entity_ids]; [key, value, qualifier_key]; [qualifier_value]; get the qualifier value of the given attribute fact, entity number should be 1
    'QueryRelationQualifier': [2, 2],   # [entity_ids_1, entity_ids_2]; [predicate, qualifier_key]; [qualifier_value]; get the qualifier value of the given relation fact, entity number should be 1
}

# program sequence -> program list
def parse_program(seq):
    program = []
    seq = seq.split(' <func> ')
    for item in seq:
        item = item.split(' <arg> ')
        func = item[0]
        inputs = item[1:]
        program.append({
            'function': func,
            'inputs': inputs
        })
    return program

# query the kg
def get_kopl_res(program, engine, pr = False):
    func_inputs = []
    for _, prog in enumerate(program):
        func_name = prog['function']
        inputs = prog['inputs']
        n_dependency, n_inputs = constrains[func_name]
        func = getattr(engine, func_name)
        
        if n_inputs == 3 and inputs[2] == '': # op
            inputs[2] = '='
        
        if n_dependency == 0:
            res = func(*inputs)
        elif n_dependency == 1:
            res = func(func_inputs[-1], *inputs)
            func_inputs.pop(-1)
        elif n_dependency == 2:
            res = func(func_inputs[-2], func_inputs[-1], *inputs)
            func_inputs.pop(-1)
            func_inputs.pop(-1)
        func_inputs.append(res)
        
    return func_inputs[-1]

# query all program sequences 
def query_kb(seqs):
    results = []
    kb = json.load(open('./data/kb.json'))
    engine = KoPLEngine(kb)
    for seq in seqs:
        program = parse_program(seq)
        print(program)
        res = get_kopl_res(program, engine)
        if isinstance(res, list):
            results.append(str(res[0]))
            # results.append([str(item)for item in res])
            # results.append([(item.value, item.unit, item.type) for item in res])
        else:
            results.append(res)
    return results

if __name__ == '__main__':
    seqs = [
        'Find <arg> FC Utrecht <func> Relate <arg> location of formation <arg> forward <func> FilterConcept <arg> big city <func> Relate <arg> named after <arg> backward <func> FilterConcept <arg> province of the Neverlands <func> Count',
        'Find <arg> FC Utrecht <func> Relate <arg> location of formation <arg> forward <func> FilterConcept <arg> big city <func> Relate <arg> named after <arg> backward <func> FilterConcept <arg> province of the Netherlands <func> Count',
        'FindAll <func> FilterNum <arg> inflation rate <arg> 4200 percentage <arg>  <func> FilterConcept <arg> unitary state <func> FindAll <func> FilterStr <arg> demonym <arg> bulgare <func> FilterConcept <arg> unitary state <func> Or <func> Count',
        'FindAll <func> FilterNum <arg> inflation rate <arg> 4200 percentage <arg>  <func> FilterConcept <arg> unitary state <func> FindAll <func> FilterStr <arg> demonym <arg> bulgare <func> FilterConcept <arg> unitary state <func> Or <func> Count',
        'FindAll <func> FilterStr <arg> CANTIC-ID <arg> a118821802 <func> FilterConcept <arg> public university <func> FindAll <func> FilterStr <arg> official website <arg> http://mq.edu.au/ <func> FilterConcept <arg> public university <func> Or <func> Count',
        'FindAll <func> FilterStr <arg> CANTIC-ID <arg> a11821802 <func> FilterConcept <arg> public university <func> FindAll <func> FilterStr <arg> official website <arg> http://mq.edu.au/ <func> FilterConcept <arg> public university <func> Or <func> Count',
        'Find <arg> New Zealand <func> QueryAttr <arg> real gross domestic product growth rate',
        'Find <arg> New Zealand <func> QueryAttr <arg> real gross domestic product growth rate',
        'FindAll <func> FilterYear <arg> date of birth <arg> 1939 <arg>  <func> FilterConcept <arg> fictional profession <func> Find <arg> Marvel Universe <func> Relate <arg> from fictional universe <arg> backward <func> FilterConcept <arg> fictional profession <func> Or <func> Count',
        'FindAll <func> FilterYear <arg> date of birth <arg> 1939 <arg>  <func> FilterConcept <arg> fictional profession <func> Find <arg> Marvel Universe <func> Relate <arg> from fictional universe <arg> backward <func> FilterConcept <arg> fictional profession <func> Or <func> Count',
        'FindAll <func> FilterYear <arg> start time <arg> 1999 <arg>  <func> FilterConcept <arg> television series <func> Count',
        'FindAll <func> FilterYear <arg> start time <arg> 1999 <arg>  <func> FilterConcept <arg> television series <func> Count',
        'Find <arg> Nikola Tesla <func> Relate <arg> field of work <arg> forward <func> FilterConcept <arg> engineering <func> FindAll <func> FilterStr <arg> IPTC newscode <arg> mediatopic/20000760 <func> FilterConcept <arg> engineering <func> Or <func> Count',
        'Find <arg> Nikola Tesla <func> Relate <arg> field of work <arg> forward <func> FilterConcept <arg> engineering <func> FindAll <func> FilterStr <arg> IPTC Newscode <arg> mediatopic/20000760 <func> FilterConcept <arg> engineering <func> Or <func> Count',
    ]
    # seqs = [
    #     'Find <arg> Nikola Tesla <func> Relate <arg> field of work <arg> forward <func> FilterConcept <arg> engineering <func> FindAll <func> FilterStr <arg> IPTC newscode <arg> mediatopic/20000760 <func> FilterConcept <arg> engineering <func> Or <func> Count',
    #     'FindAll <func> FilterYear <arg> start time <arg> 1999 <arg>  <func> FilterConcept <arg> television series <func> Count',
    #     ]
    results = query_kb(seqs)
    print(results)
