# This is the prompt without RAG
prompt_classify_one_without_memory = """
Here is a sentence from a threat report describing an attack technique.
<sentence>{text}</sentence>
Please classify the sentence into a single Technique ID (not sub-technique):
You should return in the following format:
```answer
Technique ID
```
for example:
```answer
T1003
```
"""

# This is prompt in knowledge application classify stage, all agent use this prompt. All RAG-based method use this prompt
prompt_classify_one_with_memory = """
Here is a sentence from a threat report describing a MITRE ATT&CK Technique.
<sentence>{text}</sentence>
Here is the retrieved knowledge for semantic disambiguation:
<knowledge>{memory}</knowledge>
Please classify the sentence into a single Technique ID (not sub-technique).
You should return in the following format:
```answer
Technique ID
```
for example:
```answer
T1003
```
"""

prompt_remember_old = """
Given a sentence from a threat report, create concise knowledge entries to improve future classifications. 

Input Context:
1. Target Sentence:
<sentence>{text}</sentence>

2. Official ATT&CK Definition:
<official_description>
{official_description}
</official_description>

3. Semantically Similar Cases for Reference:
<similar>
{similar}
</similar>

Task Requirements:
1. Create generalized knowledge entries that facilitate correct classification of similar cases
2. Focus on behavioral patterns that enable discrimination between semantically ambiguous techniques
3. Ensure the knowledge entries are reusable for future classifications

Knowledge Entry Structure Guidelines:
1. State Field (Shared Context):
   - Describe the shared behavioral pattern where semantic ambiguity arises
   - Keep it technique-agnostic to capture overlap between multiple techniques
   - Make it sufficiently broad to encompass various candidate techniques
   - Focus on observable threat behaviors or indicators

2. Action Field (Distinctions):
   - List relevant Technique IDs as keys
   - For each technique, specify:
     * Key distinguishing features that uniquely identify this technique under the shared context
     * Contrastive manifestations relative to semantically similar techniques
     * Critical discriminative characteristics

Output Format:
```json
{{
    "state": "Concise description of shared behavioral pattern",
    "action": {{
        "T1234": "Key distinguishing manifestation",
        "T5678": "Key distinguishing manifestation"
    }}
}}
```

Important:
- Prioritize conciseness and precision over verbose explanations
- Ensure clarity and interpretability
- Focus exclusively on the most discriminative features
- Avoid redundant examples or contextual elaborations
- Use domain-specific technical terminology
"""


# This is the prompt for reclassify
prompt_reclassify = """
This text describes a behavior corresponding to an ATT&CK Technique.
The text is: {text}
The initial classification result is: {result}
Here are distinctions from semantically ambiguous candidate techniques:
{memory}
Please perform disambiguation and classify the text into one of the following Techniques:
{Techniques}
Please answer in the following format:
```answer
{Single_Technique}
```
"""
