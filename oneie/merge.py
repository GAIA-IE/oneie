import os
import glob
import json
import itertools
from copy import deepcopy
from shutil import copyfile
from typing import List, Dict, Any, Tuple, Set, Union


def read_json_file(path: str) -> List[Dict[str, Any]]:
    """Read a JSON format output file and return a list of results.

    Args:
        path (str): path to the input JSON file.

    Returns:
        List[Dict[str, Any]]: a list of result dicts.
    """
    with open(path) as r:
        data = [json.loads(line) for line in r]
    return data


def merge_spans(old_spans: List[Tuple],
                new_spans: List[Tuple],
                special_types: Set[str] = None) -> List[Tuple]:
    """Merge two lists of spans.
    All spans produced by the M18 model are retained. A new span will be
    discarded if it overlaps with an old one.
    Entity mention span tuple format: (start_token_offset, end_token_offset,
    entity_type, mention_type, score). For example, (1, 4, 'PER', 'NOM', 1.0)
    means the span from the 1st (inclusive) token to the 4th (exclusive) token
    is a PER nominal mention with a confidence score of 1.0.
    Event trigger span tuple format: (start_token_offset, end_token_offset,
    event_type, score).
    
    Args:
        old_spans (List[Tuple]): a list of old spans produced by the M18 model.
        new_spans (List[Tuple]): a list of new spans produced by the M36 model.

    Returns:
        List[Tuple]: merged span list.
    """
    if len(old_spans) + len(new_spans) == 0:
        return []
    max_len = max([span[1] for span in itertools.chain(old_spans, new_spans)])
    
    merged_spans = []
    
    special_tokens = [0] * max_len
    if special_types is not None:
        for span in new_spans:
            start, end, eventtype = span[0], span[1], span[2]
            if eventtype in special_types:
                for idx in range(start, end):
                    special_tokens[idx] = 1
                merged_spans.append(deepcopy(span))
    
    tokens = [0] * max_len
    for span in old_spans:
        start, end = span[0], span[1]
        if any(special_tokens[idx] == 1 for idx in range(start, end)):
            continue
        for idx in range(start, end):
            tokens[idx] = 1
        merged_spans.append(span)
    
    # spans = deepcopy(old_spans)
    for span in new_spans:
        start, end = span[0], span[1]
        if all([tokens[idx] == 0 for idx in range(start, end)]):
            # spans.append(span)
            merged_spans.append(deepcopy(span))
    
    merged_spans.sort(key=lambda x: x[0])
    return merged_spans


def find_overlapping_triggers(old_spans: List[Tuple],
                              new_spans: List[Tuple]) -> Set[int]:
    """Find overlapping event triggers.
    For entities, it is acceptable that the M18 and M36 models tag the same span.
    For example, a PER entity may act as an argument of an old event and a new
    event at the same time.
    However, if an old event trigger and a new event trigger have the same span,
    we need to remove the new one.
    
    Args:
        old_spans (List[Tuple]): a list of old spans produced by the M18 model.
        new_spans (List[Tuple]): a list of new spans produced by the M36 model.
    
    Returns:
        Set[int]: indices of removed event triggers.
    """
    if len(old_spans) + len(new_spans) == 0:
        return set()
    max_len = max([span[1] for span in itertools.chain(old_spans, new_spans)])
    tokens = [0] * max_len
    for span in old_spans:
        start, end = span[0], span[1]
        for idx in range(start, end):
            tokens[idx] = 1
    
    removed_spans = set()
    for span_idx, span in enumerate(new_spans):
        start, end = span[0], span[1]
        if any([tokens[idx] == 1 for idx in range(start, end)]):
            removed_spans.add(span_idx)
    
    return removed_spans


def find_overlapping_triggers_adv(old_spans: List[Tuple],
                                  new_spans: List[Tuple],
                                  special_types: Union[Set[str], List[str]]
                                  ) -> Tuple[Set[int], Set[int]]:
    if len(old_spans) + len(new_spans) == 0:
        return set(), set()
    
    max_len = max([span[1] for span in itertools.chain(old_spans, new_spans)])
    
    # Special types
    tokens_special = [0] * max_len
    for span in new_spans:
        start, end, eventtype = span[0], span[1], span[2]
        if eventtype in special_types:
            for idx in range(start, end):
                tokens_special[idx] = 1
    
    tokens = [0] * max_len
    removed_old_spans = set()
    for span_idx, span in enumerate(old_spans):
        start, end = span[0], span[1]
        if any(tokens_special[idx] == 1 for idx in range(start, end)):
            removed_old_spans.add(span_idx)
        else:
            for idx in range(start, end):
                tokens[idx] = 1
    
    removed_new_spans = set()
    for span_idx, span in enumerate(new_spans):
        start, end = span[0], span[1]
        if any(tokens[idx] == 1 for idx in range(start, end)):
            removed_new_spans.add(span_idx)
            
    return removed_old_spans, removed_new_spans


def get_new_index(old_list, merged_list):
    """Map from indices in the old mention/trigger list to indices in the merged
    mention/trigger list.

    Args:
        old_list (List[Tuple]): old span list.
        merged_list (List[Tuple]): merged span list

    Returns:
        Dict[int, int]: a dict where the keys are old indices and values are new
        indices.
    """
    new_offset_to_idx = {(span[0], span[1]): idx
                         for idx, span in enumerate(merged_list)}
    idx_mapping = {}
    for idx, span in enumerate(old_list):
        offset = (span[0], span[1])
        if offset in new_offset_to_idx:
            idx_mapping[idx] = new_offset_to_idx[offset]
    return idx_mapping


def merge_sent_results(old_sent_data: Dict[str, Any],
                       new_sent_data: Dict[str, Any]) -> Dict[str, Any]:
    """Merge M18 and M38 IE results for the same sentence.
    Sentence-level IE result example:
    ```
    {
        "doc_id": "JC002YEPE",
        "sent_id": "JC002YEPE-6",
        "tokens": ["Yesterday", ",", "the", "president", "of", "the", "Aragua",
                   "Medical", "College", ",", "Angel", "Sarmiento", ",", "said",
                   "that", "a", "disease", "of", "unknown", "origin", "had",
                   "killed", "four", "patients", "in", "the", "hospital",
                   "since", "Monday", "."],
        "token_ids": ["JC002YEPE:419-427", "JC002YEPE:428-428",
                      "JC002YEPE:430-432", "JC002YEPE:434-442",
                      "JC002YEPE:444-445", "JC002YEPE:447-449", ...],
        "graph": {
            "entities": [[3, 4, "Person", "NOM", 1.0],
                         [6, 9, "Organization", "NAM", 0.5178251932888547],
                         [10, 12, "Person", "NAM", 1.0],
                         [23, 24, "Person", "NOM", 0.5097160857306051],
                         [26, 27, "Facility", "NOM", 1.0]],
            "triggers": [[21, 22, "Life.Die", 1.0]],
            "relations": [[0, 1, "OrganizationAffiliation", 1.0,
                           "OrganizationAffiliation.EmploymentMembership"],
                         [3, 4, "Physical", 1.0, "Physical.LocatedNear"]],
            "roles": [[0, 3, "Victim", 0.7578336249866083],
                      [0, 4, "Place", 0.7578336249866083]]}}
    ```

    Args:
        old_sent_data (Dict[str, Any]): M18 IE result.
        new_sent_data (Dict[str, Any]): M36 IE result.

    Returns:
        Dict[str, Any]: merged IE result.
    """
    old_graph, new_graph = old_sent_data['graph'], new_sent_data['graph']

    # Merge entity mentions    
    old_entities, new_entities = old_graph['entities'], new_graph['entities']
    entities = merge_spans(old_entities, new_entities)
    
    # Merge event triggers
    old_triggers, new_triggers = old_graph['triggers'], new_graph['triggers']
    triggers = merge_spans(old_triggers, new_triggers)
    removed_triggers = find_overlapping_triggers(old_triggers, new_triggers)
    
    # ap from indices in the old list to indices in the merged list
    old_entity_offset_to_idx = get_new_index(old_entities, entities)
    new_entity_offset_to_idx = get_new_index(new_entities, entities)
    old_trigger_offset_to_idx = get_new_index(old_triggers, triggers)
    new_trigger_offset_to_idx = get_new_index(new_triggers, triggers)
    
    # Merge relations
    old_relations, new_relations = old_graph['relations'], new_graph['relations']
    relations = []
    for ent_idx_1, ent_idx_2, rel_type, rel_score, rel_subtype in old_relations:
        ent_idx_1 = old_entity_offset_to_idx.get(ent_idx_1, None)
        ent_idx_2 = old_entity_offset_to_idx.get(ent_idx_2, None)
        if ent_idx_1 is not None and ent_idx_2 is not None:
            relations.append((ent_idx_1, ent_idx_2, rel_type, rel_score, rel_subtype))
    # The current M36 model actually doesn't predict any relations
    for ent_idx_1, ent_idx_2, rel_type, rel_score, rel_subtype in new_relations:
        ent_idx_1 = new_entity_offset_to_idx.get(ent_idx_1, None)
        ent_idx_2 = new_entity_offset_to_idx.get(ent_idx_2, None)
        if ent_idx_1 is not None and ent_idx_2 is not None:
            relations.append((ent_idx_1, ent_idx_2, rel_type, rel_score, rel_subtype))
    relations.sort(key=lambda x: (x[0], x[1]))
            
    # Merge roles
    old_roles, new_roles = old_graph['roles'], new_graph['roles']
    roles = []
    for trg_idx, ent_idx, role, role_score in old_roles:
        trg_idx = old_trigger_offset_to_idx.get(trg_idx, None)
        ent_idx = old_entity_offset_to_idx.get(ent_idx, None)
        if trg_idx is not None and ent_idx is not None:
            roles.append((trg_idx, ent_idx, role, role_score))
    for trg_idx, ent_idx, role, role_score in new_roles:
        if trg_idx in removed_triggers:
            continue
        trg_idx = new_trigger_offset_to_idx.get(trg_idx, None)
        ent_idx = new_entity_offset_to_idx.get(ent_idx, None)
        if trg_idx is not None and ent_idx is not None:
            roles.append((trg_idx, ent_idx, role, role_score))
    roles.sort(key=lambda x: (x[0], x[1]))
    
    merge_graph = {'entities': entities,
                   'triggers': triggers,
                   'relations': relations,
                   'roles': roles}
    return {
        'doc_id': old_sent_data['doc_id'],
        'sent_id': old_sent_data['sent_id'],
        'tokens': old_sent_data['tokens'],
        'token_ids': old_sent_data['token_ids'],
        'graph': merge_graph
    }
    

def merge_sent_results_adv(old_sent_data: Dict[str, Any],
                           new_sent_data: Dict[str, Any],
                           special_types: Union[Set[str], List[str]]) -> Dict[str, Any]:
    """Merge M18 and M38 IE results for the same sentence.
    Sentence-level IE result example:
    ```
    {
        "doc_id": "JC002YEPE",
        "sent_id": "JC002YEPE-6",
        "tokens": ["Yesterday", ",", "the", "president", "of", "the", "Aragua",
                   "Medical", "College", ",", "Angel", "Sarmiento", ",", "said",
                   "that", "a", "disease", "of", "unknown", "origin", "had",
                   "killed", "four", "patients", "in", "the", "hospital",
                   "since", "Monday", "."],
        "token_ids": ["JC002YEPE:419-427", "JC002YEPE:428-428",
                      "JC002YEPE:430-432", "JC002YEPE:434-442",
                      "JC002YEPE:444-445", "JC002YEPE:447-449", ...],
        "graph": {
            "entities": [[3, 4, "Person", "NOM", 1.0],
                         [6, 9, "Organization", "NAM", 0.5178251932888547],
                         [10, 12, "Person", "NAM", 1.0],
                         [23, 24, "Person", "NOM", 0.5097160857306051],
                         [26, 27, "Facility", "NOM", 1.0]],
            "triggers": [[21, 22, "Life.Die", 1.0]],
            "relations": [[0, 1, "OrganizationAffiliation", 1.0,
                           "OrganizationAffiliation.EmploymentMembership"],
                         [3, 4, "Physical", 1.0, "Physical.LocatedNear"]],
            "roles": [[0, 3, "Victim", 0.7578336249866083],
                      [0, 4, "Place", 0.7578336249866083]]}}
    ```

    Args:
        old_sent_data (Dict[str, Any]): M18 IE result.
        new_sent_data (Dict[str, Any]): M36 IE result.

    Returns:
        Dict[str, Any]: merged IE result.
    """
    old_graph, new_graph = old_sent_data['graph'], new_sent_data['graph']

    # Merge entity mentions    
    old_entities, new_entities = old_graph['entities'], new_graph['entities']
    entities = merge_spans(old_entities, new_entities)
    
    # Merge event triggers
    old_triggers, new_triggers = old_graph['triggers'], new_graph['triggers']
    triggers = merge_spans(old_triggers, new_triggers, special_types)
    removed_old_triggers, removed_new_triggers = find_overlapping_triggers_adv(old_triggers, new_triggers, special_types)
    
    # ap from indices in the old list to indices in the merged list
    old_entity_offset_to_idx = get_new_index(old_entities, entities)
    new_entity_offset_to_idx = get_new_index(new_entities, entities)
    old_trigger_offset_to_idx = get_new_index(old_triggers, triggers)
    new_trigger_offset_to_idx = get_new_index(new_triggers, triggers)
    
    # Merge relations
    old_relations, new_relations = old_graph['relations'], new_graph['relations']
    relations = []
    for ent_idx_1, ent_idx_2, rel_type, rel_score, rel_subtype in old_relations:
        ent_idx_1 = old_entity_offset_to_idx.get(ent_idx_1, None)
        ent_idx_2 = old_entity_offset_to_idx.get(ent_idx_2, None)
        if ent_idx_1 is not None and ent_idx_2 is not None:
            relations.append((ent_idx_1, ent_idx_2, rel_type, rel_score, rel_subtype))
    # The current M36 model actually doesn't predict any relations
    for ent_idx_1, ent_idx_2, rel_type, rel_score, rel_subtype in new_relations:
        ent_idx_1 = new_entity_offset_to_idx.get(ent_idx_1, None)
        ent_idx_2 = new_entity_offset_to_idx.get(ent_idx_2, None)
        if ent_idx_1 is not None and ent_idx_2 is not None:
            relations.append((ent_idx_1, ent_idx_2, rel_type, rel_score, rel_subtype))
    relations.sort(key=lambda x: (x[0], x[1]))
            
    # Merge roles
    old_roles, new_roles = old_graph['roles'], new_graph['roles']
    roles = []
    for trg_idx, ent_idx, role, role_score in old_roles:
        if trg_idx in removed_old_triggers:
            continue
        trg_idx = old_trigger_offset_to_idx.get(trg_idx, None)
        ent_idx = old_entity_offset_to_idx.get(ent_idx, None)
        if trg_idx is not None and ent_idx is not None:
            roles.append((trg_idx, ent_idx, role, role_score))
    for trg_idx, ent_idx, role, role_score in new_roles:
        if trg_idx in removed_new_triggers:
            continue
        trg_idx = new_trigger_offset_to_idx.get(trg_idx, None)
        ent_idx = new_entity_offset_to_idx.get(ent_idx, None)
        if trg_idx is not None and ent_idx is not None:
            roles.append((trg_idx, ent_idx, role, role_score))
    roles.sort(key=lambda x: (x[0], x[1]))
    
    merge_graph = {'entities': entities,
                   'triggers': triggers,
                   'relations': relations,
                   'roles': roles}
    return {
        'doc_id': old_sent_data['doc_id'],
        'sent_id': old_sent_data['sent_id'],
        'tokens': old_sent_data['tokens'],
        'token_ids': old_sent_data['token_ids'],
        'graph': merge_graph
    }
    
    
def merge_doc_results(old_data: List[Dict[str, Any]],
                      new_data: List[Dict[str, Any]],
                      special_types: Set[str]) -> List[Dict[str, Any]]:
    """Merge M18 and M38 IE results for the same document.

    Args:
        old_data (List[Dict[str, Any]]): a list of old IE results.
        new_data (List[Dict[str, Any]]): a list of new IE results.

    Returns:
        List[Dict[str, Any]]: a list of merged IE results.
    """
    old_data = {rst['sent_id']: rst for rst in old_data}
    new_data = {rst['sent_id']: rst for rst in new_data}
    # sent_ids = list(set([sent_id for sent_id in itertools.chain(old_data, new_data)]))
    # sent_ids.sort()
    sent_ids = [sent_id for sent_id in old_data]
    for sent_id in new_data:
        if sent_id not in sent_ids:
            sent_ids.add(sent_id)

    merge_data = []
    for sent_id in sent_ids:
        old_sent_data = old_data.get(sent_id, None)
        new_sent_data = new_data.get(sent_id, None)
        if old_sent_data is None:
            merge_data.append(new_sent_data)
        elif new_sent_data is None:
            merge_data.append(old_sent_data)
        else:
            # merge_data.append(merge_sent_results(old_sent_data, new_sent_data))
            merge_data.append(merge_sent_results_adv(old_sent_data,
                                                     new_sent_data,
                                                     special_types))
    return merge_data


def merge_json_result(old_json_path: str,
                      new_json_path: str,
                      output_path: str,
                      special_types: Set[str]):
    """Merge JSON format result.
    The IE result of each document is saved to a separate JSON file in the input
    directories, where each line in the file represents the IE result of a 
    sentence in the document.

    Args:
        old_json_path (str): path to the old JSON format result directory.
        new_json_path (str): path to the new JSON format result directory.
        output_path (str): path to the output directory.
    """
    old_json_files = glob.glob(os.path.join(old_json_path, '*.json'))
    old_doc_ids = [os.path.basename(f).replace('.json', '')
                   for f in old_json_files]
    # new_json_files = glob.glob(os.path.join(new_json_path, '*.json'))
    new_doc_ids = [os.path.basename(f).replace('.json', '')
                   for f in old_json_files]

    # Get all doc IDs
    doc_ids = set(old_doc_ids + new_doc_ids)

    # Merge JSON result
    for doc_id in doc_ids:
        old_json_file = os.path.join(old_json_path, '{}.json'.format(doc_id))
        new_json_file = os.path.join(new_json_path, '{}.json'.format(doc_id))
        output_file = os.path.join(output_path, '{}.json'.format(doc_id))
        if not os.path.exists(old_json_file):
            copyfile(new_json_file, output_file)
        elif not os.path.exists(new_json_file):
            copyfile(old_json_file, output_file)
        else:
            old_result = read_json_file(old_json_file)
            new_result = read_json_file(new_json_file)
            merged_result = merge_doc_results(old_result, new_result, special_types)
            with open(output_file, 'w') as w:
                for rst in merged_result:
                    w.write(json.dumps(rst) + '\n')


def merge_hidden_state(old_path: str,
                       new_path: str,
                       output_path: str):
    """Merge hidden state outputs.

    Args:
        old_path (str): path to the old hidden state file.
        new_path (str): path to the new hidden state file.
        output_path (str): path the merged hidden state file.
    """
    with open(output_path, 'w') as w:
        seen_span_set = set()
        with open(old_path) as r:
            for line in r:
                span = line.split('\t')[1]
                seen_span_set.add(span)
                w.write(line)
        with open(new_path) as r:
            for line in r:
                span = line.split('\t')[1]
                if span not in seen_span_set:
                    w.write(line)


# if __name__ == '__main__':
    
#     old_file = '/shared/nas/data/m1/yinglin8/aida/result/dryrun_e11_en_aug_27/m18/json/KC003ADR9.json'
#     new_file = '/shared/nas/data/m1/yinglin8/aida/result/dryrun_e11_en_aug_27/m36/json/KC003ADR9.json'
#     old_result = read_json_file(old_file)
#     new_result = read_json_file(new_result)
#     merge_result = merge_doc_results(old_result, new_resule)
#     merge_file = '/shared/nas/data/m1/yinglin8/tmpfile/oneie_debug/'