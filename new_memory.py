import json
from collections import defaultdict
import re

# with open("./records/memories_procedures_orign_32B_10.json", "r") as f:
#     memories = json.load(f)
with open("./records/memories_procedures_deepseekchat.json", "r") as f:
    memories = json.load(f)


action_to_labels = defaultdict(list)

action_to_actions = defaultdict(list)
actions = action_to_labels.keys()

for memory in memories:
    actions = memory['action']
    for label, action in actions.items():
        action_to_labels[action].append(label)
        action_to_actions[action].append(actions)

new_memories = []
for state, actions in action_to_actions.items():
    new_memory = {
        "state": state,
        "action": {key: value for item in actions for key, value in item.items()}
    }
    new_memories.append(new_memory)


with open("./records/memories_procedures_deepseekchat_new.json", "w") as f:
    json.dump(new_memories, f, indent=4)
