import g2o

class VertexSE3WithTime(g2o.VertexSE3):
    def __init__(self):
        super().__init__()
        self.timestamp = None  # Add a timestamp field

    def set_timestamp(self, timestamp):
        self.timestamp = timestamp

    def get_timestamp(self):
        return self.timestamp
