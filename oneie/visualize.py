import os
import json
from yattag import Doc, indent
from typing import Dict, List, Tuple, Any
from collections import defaultdict


class Style(object):
    
    def __init__(self):
        self.styles = defaultdict(dict)
    
    def add_style(self, selector: str, styles: Dict[str, Any]):
        for k, v in styles.items():
            self.styles[selector][k] = v 
    
    def html(self):
        style_list = []
        for selector, styles in self.styles.items():
            styles = ';'.join('{}: {}'.format(k, v) for k, v in styles.items())
            style_list.append(selector + '{' + styles + '}')
        return '<style>\n' + '\n'.join(style_list) + '\n</style>'   
    

style = Style()
style.add_style('html',
                {'font-family': 'Roboto, Helvetica, Arial, sans-serif',
                 'font-size': '14px',
                 'line-height': '1.4',
                 'color': '#333',
                 'padding': '10px',
                 'padding-top': '20px'})
style.add_style('.inst',
                {'background-color': '#fff',
                 'box-shadow': '0 0 10px rgba(0,0,0,.05)',
                 'padding': '15px',
                 'margin-bottom': '10px',
                 'border': '1px solid #eee'})
style.add_style('.mention_token',
                {'color': 'rgb(92, 124, 167)'})
style.add_style('.trigger_token',
                {'color': 'rgb(215, 32, 163)'})
style.add_style('.sent_id',
                {'color': '#ccc',
                 'font-size': '11px'})
style.add_style('.entity, .relation, .event',
                {'font-size': '12px'})
style.add_style('ul',
                {'padding': 0,
                 'padding-left': '20px',
                 'list-style-type': 'decimal'})
style.add_style('.title',
                {'font-size': '12px',
                 'text-align': 'center',
                 'font-weight': 600,
                 'margin-bottom': '10px'})
style.add_style('.side[title="predicted"]',
                {'border-left': '1px solid #eee'})

def bootstrap(doc: Doc):
    tag = doc.tag
    
    doc.stag('link',
             rel='stylesheet',
             href='https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css',
             integrity="sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T",
             crossorigin="anonymous")
    with tag('script',
             src='https://code.jquery.com/jquery-3.4.1.min.js',
             integrity="sha256-CSXorXvZcTkaix6Yvo6HppcZGetbYMGWSFlBw8HfCJo=",
             crossorigin="anonymous"): pass
    with tag('script',
             src='https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.14.7/umd/popper.min.js',
             integrity="sha384-UO2eT0CpHqdSJQ6hJty5KVphtPhzWj9WO1clHTMGa3JDZwrnQq4sF86dIHNDz0W1",
             crossorigin="anonymous"): pass
    with tag('script',
             src='https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/js/bootstrap.min.js',
             integrity="sha384-JjSmVgyd0p3pXB1rRibZUAYoIIy6OrQ6VrjIEaFf/nJGzIxFDsf4x0xIM+B07jRM",
             crossorigin="anonymous"): pass
    
def visualize_side(tokens, graph, doc, title):
    doc, tag, text, line = doc.ttl()
    
    entities = graph['entities']
    relations = graph['relations']
    triggers = graph['triggers']
    roles = graph['roles']
    
    # Mark tokens in spans
    mention_tokens = [False] * len(tokens)
    trigger_tokens = [False] * len(tokens)
    for entity in entities:
        for i in range(entity[0], entity[1]):
            mention_tokens[i] = True
    for trigger in triggers:
        for i in range(trigger[0], trigger[1]):
            trigger_tokens[i] = True
    
    with tag('div', klass='side col-md-6', title=title.lower()):
        line('div', title, klass='title')
        for token, is_mention, is_trigger in zip(tokens, mention_tokens, trigger_tokens):
            with tag('span',
                     klass='token {}'.format('mention_token' if is_mention
                                             else 'trigger_token' if is_trigger
                                             else '')):
                text(token)
            text(' ')
            
        if entities or relations or triggers:
            doc.stag('hr')
        
        if entities:
            line('span', 'Entities:')
            with tag('ul', klass='entity-wrapper'):
                for entity in entities:
                    entity_text = ' '.join(tokens[entity[0]:entity[1]])
                    with tag('li', klass='entity', start=entity[0], end=entity[1]):
                        with tag('span'):
                            entity_span = str(entity[0] + 1) if entity[0] + 1 == entity[1] \
                                else '{}:{}'.format(entity[0] + 1, entity[1])
                            line('span', entity_text)
                            text(': ')
                            line('span', entity[2] + ' [{}]'.format(entity_span))
            
        if relations:
            line('span', 'Relations:')
            with tag('ul', klass='relation-wrapper'):
                for relation in relations:
                    entity_1 = entities[relation[0]]
                    entity_2 = entities[relation[1]]
                    entity_text_1 = ' '.join(tokens[entity_1[0]:entity_1[1]])
                    entity_text_2 = ' '.join(tokens[entity_2[0]:entity_2[1]])
                    with tag('li', klass='relation',
                             start_1=entities[relation[0]][0],
                             end_1=entities[relation[0]][1],
                             start_2=entities[relation[1]][0],
                             end_2=entities[relation[1]][1]):
                        line('span',
                             entity_text_1 + ' ↔︎ ' + entity_text_2 + ': ' + relation[2])
                    
            
        if triggers:
            line('span', 'Events')
            with tag('ul', klass='event-wrapper'):
                for trigger_idx, trigger in enumerate(triggers):
                    trigger_text = ' '.join(tokens[trigger[0]:trigger[1]])
                    with tag('li', klass='event', start=trigger[0], end=trigger[1]):
                        with tag('span'):
                            line('span', trigger_text)
                            
                            # text(' [{}:{}]: '.format(trigger[0] + 1, trigger[1]))
                            trigger_span = str(trigger[0] + 1) if trigger[0] + 1 == trigger[1]\
                                else '{}:{}'.format(trigger[0] + 1, trigger[1])
                            text(' [{}]: '.format(trigger_span))
                            
                            line('span', trigger[2])
                            args = []
                            for role in roles:
                                trigger_, entity_, role_ = role
                                if trigger_ == trigger_idx:
                                    args.append((entity_, role_))
                            if args:
                                with tag('ul'):
                                    for entity_, role_ in args:
                                        entity_text = ' '.join(tokens[entities[entity_][0]:
                                                                      entities[entity_][1]])
                                        line('li',
                                             '{}: {} [{}]'.format(role_, entity_text, entity_ + 1))
    

def visualize_instance_compare(inst: Dict[str, Any], doc: Doc):
    doc, tag, text, line = doc.ttl()
    
    sent_id = inst['sent_id']
    tokens = inst['tokens']
    
    gold = inst['gold']
    pred = inst['pred']
    
    with tag('div', klass='inst container-fluid'):
        line('div', sent_id, klass='sent_id')
        with tag('div', klass='row'):
            visualize_side(tokens, gold, doc, 'Ground Truth')
            visualize_side(tokens, pred, doc, 'Predicted')
    

def visualize_instance(inst: Dict[str, Any], doc: Doc):
    doc, tag, text, line = doc.ttl()
    
    doc_id = inst['doc_id']
    sent_id = inst['sent_id']
    token_ids = inst['token_ids']
    tokens = inst['tokens']
    graph = inst['graph']
    entities = graph['entities']
    relations = graph['relations']
    triggers = graph['triggers']
    roles = graph['roles']
    
    # Mark tokens in spans
    mention_tokens = [False] * len(tokens)
    trigger_tokens = [False] * len(tokens)
    for entity in entities:
        for i in range(entity[0], entity[1]):
            mention_tokens[i] = True
    for trigger in triggers:
        for i in range(trigger[0], trigger[1]):
            trigger_tokens[i] = True
    
    with tag('div', klass='inst'):
        line('div', sent_id, klass='sent_id')
        for token, token_id, is_mention, is_trigger in zip(tokens,
                                                           token_ids,
                                                           mention_tokens,
                                                           trigger_tokens
                                                           ):
            with tag('span',
                     klass='token {}'.format('mention_token' if is_mention
                                             else 'trigger_token' if is_trigger
                                             else ''), 
                     token_id=token_id):
                text(token)
            text(' ')
            
        if entities or relations or triggers:
            doc.stag('hr')
        
        if entities:
            line('span', 'Entities:')
            with tag('ul', klass='entity-wrapper'):
                for entity in entities:
                    entity_text = ' '.join(tokens[entity[0]:entity[1]])
                    with tag('li', klass='entity', start=entity[0], end=entity[1]):
                        with tag('span'):
                            line('span', entity_text)
                            text(': ')
                            line('span', entity[2] + ', ' + entity[3])
            
        if relations:
            line('span', 'Relations:')
            with tag('ul', klass='relation-wrapper'):
                for relation in relations:
                    entity_1 = entities[relation[0]]
                    entity_2 = entities[relation[1]]
                    entity_text_1 = ' '.join(tokens[entity_1[0]:entity_1[1]])
                    entity_text_2 = ' '.join(tokens[entity_2[0]:entity_2[1]])
                    with tag('li', klass='relation',
                             start_1=entities[relation[0]][0],
                             end_1=entities[relation[0]][1],
                             start_2=entities[relation[1]][0],
                             end_2=entities[relation[1]][1]):
                        line('span',
                             entity_text_1 + ' ↔︎ ' + entity_text_2 + ': ' + relation[2])
                    
            
        if triggers:
            line('span', 'Events')
            with tag('ul', klass='event-wrapper'):
                for trigger_idx, trigger in enumerate(triggers):
                    trigger_text = ' '.join(tokens[trigger[0]:trigger[1]])
                    with tag('li', klass='event', start=trigger[0], end=trigger[1]):
                        with tag('span'):
                            line('span', trigger_text)
                            text(': ')
                            line('span', trigger[2])
                            args = []
                            for role in roles:
                                trigger_, entity_, role_, _ = role
                                if trigger_ == trigger_idx:
                                    args.append((entity_, role_))
                            if args:
                                with tag('ul'):
                                    for entity_, role_ in args:
                                        entity_text = ' '.join(tokens[entities[entity_][0]:
                                                                      entities[entity_][1]])
                                        line('li',
                                             '{}: {}'.format(role_, entity_text))


def visualize(input_file: str,
              output_file: str):
    doc, tag, text, line = Doc().ttl()
    
    with tag('html'):
        with tag('head'):
            doc.stag('meta', charset='UTF-8')
            bootstrap(doc)
            doc.asis(style.html())
        with tag('body'):
            with open(input_file) as r:
                for line in r:
                    inst = json.loads(line)
                    visualize_instance_compare(inst, doc)
    
    with open(output_file, 'w') as w:
        w.write(indent(doc.getvalue()))
        
        
visualize(
    # '/shared/nas/data/m1/yinglin8/projects/oneie/result/kairos_brat/20201003_181651/result.dev.json',
    '/shared/nas/data/m1/yinglin8/projects/oneie/result/ere_english/20201020_105753/result.dev.json',
    '/shared/nas/data/m1/yinglin8/projects/oneie/result/ere_english/20201020_105753/dev.html'
)