import pickle

import substratools as tools


class AlgoGlobal(tools.algo.AggregateAlgo):

    def aggregate(self, models, rank):
        training_num = 0
        for idx in range(len(models)):
            # Discard the second field of the tuple (model_params)
            sample_num, _ = models[idx]
            training_num += sample_num

        _, averaged_params = models[0]
        for k in averaged_params.keys():
            for i in range(0, len(models)):
                sample_num, model_params = models[i]
                w = sample_num / training_num
                if i == 0:
                    averaged_params[k] = model_params[k] * w
                else:
                    averaged_params[k] += model_params[k] * w

        return training_num, averaged_params

    def load_model(self, path):
        sample_num = 0
        with open(path, 'rb') as f:
            sample_num, model_params = pickle.load(f)
        return sample_num, model_params

    def save_model(self, model, path):
        with open(path, 'wb') as f:
            pickle.dump(model, f)


if __name__ == '__main__':
    tools.algo.execute(AlgoGlobal())
