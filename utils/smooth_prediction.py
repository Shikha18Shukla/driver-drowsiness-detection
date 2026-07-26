from collections import deque

class PredictionSmoother:
    def __init__(self, window_size=5, drowsy_threshold=0.6, awake_threshold=0.7, consecutive_required=3):
        self.predictions = deque(maxlen=window_size)
        self.state = 'AWAKE'
        self.drowsy_count = 0
        self.awake_count = 0
        self.consecutive_required = consecutive_required
        self.drowsy_threshold = drowsy_threshold
        self.awake_threshold = awake_threshold

    def update(self, prob):
        self.predictions.append(prob)
        avg_prob = sum(self.predictions) / len(self.predictions)

        if avg_prob < self.drowsy_threshold:
            self.drowsy_count += 1
            self.awake_count = 0
        elif avg_prob > self.awake_threshold:
            self.awake_count += 1
            self.drowsy_count = 0
        else:
            self.drowsy_count = 0
            self.awake_count = 0

        if self.drowsy_count >= self.consecutive_required:
            self.state = 'DROWSY'
        elif self.awake_count >= self.consecutive_required:
            self.state = 'AWAKE'

        return self.state, avg_prob
