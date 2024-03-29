OneIE v0.3

Requirements
============
Python 3.5+
Python packages
- PyTorch 1.0+ (Install the CPU version if you use this tool on a machine without GPUs)
- transformers
- tqdm
- lxml


How to Run
==========
- `cd` to the root directory of this package
- Set the environment variable PYTHONPATH to the current directory.
  For example, if you unpack this package to `~/oneie_v0.3`, run:
  `export PYTHONPATH=~/oneie_v0.3`
- Example commandline to use OneIE: `python predict.py -m best.role.mdl -i input -o output -c output_cs`
  + Arguments:
    - -m, --model_path: Path to the trained model. With this package I include a model file `best.role.mdl`.
    - -i, --input_dir: Path to the input directory. File format is LTF. Sample files can be found in the `input` directory.
    - -o, --output_dir: Path to the output directory (json format). Output files are in the JSON format. Sample files can be found in the `output` directory.
    - -c, --cs_dir: (optional) Path to the output directory (cs format). Sample files can be found in the `output_cs` directory.
    - -l, --log_path: (optional) Path to the log file. A sample file `log.json` can be found in `output`.
    - --gpu: (optional) Use GPU
    - -d, --device: (optional) GPU device index (for multi-GPU machines).
    - -b, --batch_size: (optional) Batch size. For a 16GB GPU, a batch size of 10~15 is a reasonable value.
    - --max_len: (optional) Max sentence length. Sentences longer than this value will be ignored. You may need to decrease `batch_size` if you set `max_len` to a larger number.
    - --beam_size: (optional) Beam set size of the decoder. Increasing this value may improve the results and make the decoding slower.
    - --lang: (optional) Model language.
    - --format: Input file format (txt or ltf).


Output Format
=============
OneIE save results in JSON format. Each line is a JSON object for a sentence 
containing the following fields:
+ doc_id (string): Document ID
+ sent_id (string): Sentence ID
+ tokens (list): A list of tokens
+ token_ids (list): A list of token IDs (doc_id:start_offset-end_offset)
+ graph (object): Information graph predicted by the model
  - entities (list): A list of predicted entities. Each item in the list has exactly
  four values: start_token_index, end_token_index, entity_type, mention_type, score.
  For example, "[3, 5, "GPE", "NAM", 1.0]" means the index of the start token is 3, 
  index of the end token is 4 (5 - 1), entity type is GPE, mention type is NAM,
  and local score is 1.0.
  - triggers (list): A list of predicted triggers. It is similar to `entities`, while
  each item has three values: start_token_index, end_token_index, event_type, score.
  - relations (list): A list of predicted relations. Each item in the list has
  three values: arg1_entity_index, arg2_entity_index, relation_type, score.
  In the following example, `[1, 0, "ORG-AFF", 0.52]` means there is a ORG-AFF relation
  between entity 1 ("leader") and entity 0 ("North Korean") with a local
  score of 0.52.
  The order of arg1 and arg2 can be ignored for "SOC-PER" as this relation is 
  symmetric.
  - roles (list): A list of predicted argument roles. Each item has three values:
  trigger_index, entity_index, role, score.
  In the following example, `[0, 2, "Attacker", 0.8]` means entity 2 (Kim Jong Un) is
  the Attacker argument of event 0 ("detonate": Conflict:Attack), and the local
  score is 0.8.

Output example:
```
{"doc_id": "HC0003PYD", "sent_id": "HC0003PYD-16", "token_ids": ["HC0003PYD:2295-2296", "HC0003PYD:2298-2304", "HC0003PYD:2305-2305", "HC0003PYD:2307-2311", "HC0003PYD:2313-2318", "HC0003PYD:2320-2325", "HC0003PYD:2327-2329", "HC0003PYD:2331-2334", "HC0003PYD:2336-2337", "HC0003PYD:2339-2348", "HC0003PYD:2350-2351", "HC0003PYD:2353-2360", "HC0003PYD:2362-2362", "HC0003PYD:2364-2367", "HC0003PYD:2369-2376", "HC0003PYD:2378-2383", "HC0003PYD:2385-2386", "HC0003PYD:2388-2390", "HC0003PYD:2392-2397", "HC0003PYD:2399-2401", "HC0003PYD:2403-2408", "HC0003PYD:2410-2412", "HC0003PYD:2414-2415", "HC0003PYD:2417-2425", "HC0003PYD:2427-2428", "HC0003PYD:2430-2432", "HC0003PYD:2434-2437", "HC0003PYD:2439-2441", "HC0003PYD:2443-2447", "HC0003PYD:2449-2450", "HC0003PYD:2452-2454", "HC0003PYD:2456-2464", "HC0003PYD:2466-2472", "HC0003PYD:2474-2480", "HC0003PYD:2481-2481", "HC0003PYD:2483-2485", "HC0003PYD:2487-2491", "HC0003PYD:2493-2502", "HC0003PYD:2504-2509", "HC0003PYD:2511-2514", "HC0003PYD:2516-2523", "HC0003PYD:2524-2524"], "tokens": ["On", "Tuesday", ",", "North", "Korean", "leader", "Kim", "Jong", "Un", "threatened", "to", "detonate", "a", "more", "powerful", "H-bomb", "in", "the", "future", "and", "called", "for", "an", "expansion", "of", "the", "size", "and", "power", "of", "his", "country's", "nuclear", "arsenal", ",", "the", "state", "television", "agency", "KCNA", "reported", "."], "graph": {"entities": [[3, 5, "GPE", "NAM", 1.0], [5, 6, "PER", "NOM", 0.2], [6, 9, "PER", "NAM", 0.5060472888322202], [15, 16, "WEA", "NOM", 0.5332313915378754], [30, 31, "PER", "PRO", 1.0], [32, 33, "WEA", "NOM", 1.0], [33, 34, "WEA", "NOM", 0.5212696155645499], [36, 37, "GPE", "NOM", 0.4998288792916457], [38, 39, "ORG", "NOM", 1.0], [39, 40, "ORG", "NAM", 0.5294904130032032]], "triggers": [[11, 12, "Conflict:Attack", 1.0]], "relations": [[1, 0, "ORG-AFF", 1.0]], "roles": [[0, 2, "Attacker", 0.4597024700555278], [0, 3, "Instrument", 1.0]]}}
```