import json
from typing import Optional

class Conversation(object):
    def __init__(self):
        self.dialogs = []
        self.meta = {}
        self.chat_template_applied = False
        
    def append(self, role, message: str, meta:Optional[dict]=None):
        dialog = {
            "role": role,
            "content": message
        }
        if meta:
            dialog.update(meta)
        self.dialogs.append(dialog)
        
    def add_meta(self, meta: dict):
        self.meta.update(meta)
    
    def sanity_check(self):
        # check if there are repeated roles in the self.dialogs
        roles = [dialog['role'] for dialog in self.dialogs]
        # check if there're two consecutive roles
        for i in range(len(roles)-1):
            if roles[i] == roles[i+1]:
                print(f"Two consecutive roles detected at index {i}, roles[i]: {roles[i]}, roles[i+1]: {roles[i+1]}")
                return False
        return True
    
    def to_dict(self):
        if not self.chat_template_applied:
            return {
                "conversations": self.dialogs,
                "meta": self.meta
            }
        else:
            return {
                "text": self.text,
                "meta": self.meta
            }
    
    def to_json_str(self):
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(self, data: dict):
        conv = Conversation()
        conv.dialogs = data["conversations"]
        conv.meta = data['meta']
        return conv
    
    def apply_chat_template(self, tokenizer):
        self.text = tokenizer.apply_chat_template(self.dialogs, tokenize=False)
        self.chat_template_applied = True
        