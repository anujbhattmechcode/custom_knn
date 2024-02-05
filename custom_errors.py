class FeatureLabelSizeMissmatch(Exception):
    
    def __init__(self, message="Size of features and labels are not same!"):
        self.message = message
    
    def __repr__(self):
        return self.message
    
    def __str__(self):
        return self.message

class LabelNotFound(Exception):
    def __init__(self, message="Cannot found the label in given dataset!"):
        self.message = message
    
    def __repr__(self):
        return self.message
    
    def __str__(self):
        return self.message
    
class InsufficientData(Exception):
    def __init__(self, message="Given data is insufficient to carry the operation"):
        self.message = message
    
    def __repr__(self):
        return self.message
    
    def __str__(self):
        return self.message