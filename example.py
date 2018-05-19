from Final_Project.utils.dataset import Dataset
from Final_Project.utils.baseline import Baseline
from Final_Project.utils.scorer import report_score


def execute_demo(language):
    data = Dataset(language)

    # print("{}: {} training - {} dev".format(language, len(data.trainset), len(data.devset)))

    # for sent in data.trainset:
    #    print(sent['sentence'], sent['target_word'], sent['gold_label'])

    baseline = Baseline(language)

    baseline.train(data.trainset)

    predictions_dev = baseline.test(data.devset)

    predictions_test = baseline.test(data.testset)

    gold_labels_dev = [sent['gold_label'] for sent in data.devset]
    gold_labels_test = [sent['gold_label'] for sent in data.testset]

    print("DEV result:")
    report_score(gold_labels_dev, predictions_dev, detailed=True)

    print("TEST result:")
    report_score(gold_labels_test, predictions_test, detailed=True)


if __name__ == '__main__':
    execute_demo('english')
    execute_demo('spanish')
