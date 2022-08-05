class Posting:
    def __init__(self, docid, tf):
        self.docid = docid
        self.tf = tf

    def getDocId(self):
        return self.docid

    def gettf(self):
        return self.tf
