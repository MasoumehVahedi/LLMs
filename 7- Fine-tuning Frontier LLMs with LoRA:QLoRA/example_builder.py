"""
   Below is a fully generic “Example” class you can drop into any fine-tuning pipeline. It wraps:

    1- Field selection (you tell it which keys hold your input text and target): Any Python dict you pulled from CSV, JSON, a HF Dataset, etc.
    2- Cleaning (pass in any number of cleaning functions)
    3- Length filtering (min/max tokens or characters)
    4- Prompt templating (you supply the {input} → {target} format)

"""




from typing import Optional, Any, Dict, Callable, List
from transformers import PreTrainedTokenizer


class TrainingExample:
    """
        A generic training example builder for fine-tuning.

        Args:
          raw               : one raw row (dict-like)
          input_key         : which key in raw holds the text to feed the model
          target_key        : which key in raw holds the label/output
          clean_funcs       : list of functions f(str) -> str to apply to input text
          tokenizer         : a HuggingFace tokenizer for token-count filters
          template          : prompt template with `{input}` and `{target}` placeholders
          min_tokens, max_tokens: discard examples outside this token range
          min_chars         : discard examples shorter than this
    """
    def __init__(self,
                 raw: Dict[str, Any],
                 input_key: str,
                 target_key: str,
                 clean_funcs: Optional[List[Callable[[str], str]]] = None,
                 tokenizer: Optional[PreTrainedTokenizer] = None,
                 template: str = "{input}\n\nAnswer: {target}",
                 min_tokens: int = 1,
                 max_tokens: int = 1000,
                 min_chars: int = 1
                 ):
        self.raw = raw
        self.input_key = input_key
        self.target_key = target_key
        self.clean_funcs = clean_funcs or []
        self.tokenizer = tokenizer
        self.template = template
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.min_chars = min_chars

        # Will be set in validate()
        self.token_count = 0

        # 1) Extract & clean the input text
        self.input_text = self.scrub(raw.get(input_key, ""))
        self.target_text = str(raw.get(target_key, ""))

        # 2) Validate lengths & token counts
        self.valid = self.validate()

        # 3) Build final prompt if valid
        self.prompt = self.buildPrompt() if self.valid else None


    def scrub(self, text: str) -> str:
        """Apply all cleaning functions in sequence."""
        for fn in self.clean_funcs:
          text = fn(text)
        return text.strip()


    def validate(self) -> bool:
        """Check length and token-count thresholds."""
        if len(self.input_text) < self.min_chars:
          return False
        if self.tokenizer:
          tokens = self.tokenizer.encode(self.input_text, add_special_tokens=False)
          self.token_count = len(tokens)
          if self.token_count < self.min_tokens:
            return False
        return True


    def buildPrompt(self):
        """Fill in the template with cleaned input and target."""
        return self.template.format(input=self.input_text, target=self.target_text)

    def toDict(self) -> Dict[str, str]:
        """
            Return the ready-to-write JSONL record:
            {"prompt": ..., "completion": ...}
        """
        return {
            "prompt": self.prompt,
            "completion": self.target_text
        }

    def testPrompt(self, prefix: str) -> str:
        """
        Strip off the {target} *and* its formatting prefix.
        """
        # split on the prefix itself instead of the suffix
        return self.prompt.split(prefix)[0] + prefix