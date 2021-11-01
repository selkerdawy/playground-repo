import sys
import transformers

def bert_base_cased():
    return transformers.AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)

def prajjwal1_bert_tiny():
    return transformers.AutoModelForSequenceClassification.from_pretrained("prajjwal1/bert-tiny", num_labels=2)

thismodule = sys.modules[__name__]
setattr(thismodule, 'bert-base-cased', bert_base_cased)
setattr(thismodule, 'prajjwal1/bert-tiny', prajjwal1_bert_tiny)