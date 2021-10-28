import substratools as tools


class MNISTMetrics(tools.Metrics):

    def score(self, y_true, y_pred):
        """Returns the accuracy of the model

        :param y_true: actual values from test data
        :type y_true: pd.DataFrame
        :param y_true: predicted values from test data
        :type y_pred: pd.DataFrame
        :rtype: float
        """

        metrics = {
            'correct': 0,
            'total': 0,
            'accuracy': 0.0
        }

        for batch_idx, (y, target) in enumerate(zip(y_pred, y_true)):
            # _, predicted = torch.max(pred, -1)
            correct = y.eq(target).sum()
            metrics['correct'] += correct.item()
            metrics['total'] += target.size(0)

        metrics['accuracy'] = metrics['correct'] / metrics['total']

        return metrics['accuracy']


if __name__ == '__main__':
    tools.metrics.execute(MNISTMetrics())
