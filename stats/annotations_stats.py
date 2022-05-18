from matplotlib import pyplot as plt
from collections import Counter


class Stats:
    def __init__(self, usage_loader):
        self.usage_loader = usage_loader

    def number_of_annotation_by_target(self):
        number_of_usage = {}

        for target, usages in self.usage_loader.usages_by_target.items():
            number_of_usage[target] = len(usages)

        targets = []
        values = []

        for target in sorted(number_of_usage, key=number_of_usage.get, reverse=True):
            targets += [target]
            values += [number_of_usage[target]]

        pl = plt.barh(targets, values, log=True)
        for bar in pl:
            width = bar.get_width()
            label_y = bar.get_y() + bar.get_height() / 2
            plt.text(width, label_y, s=f'{width}')

        plt.xlabel('number of annotations', fontsize=15)
        plt.show()

    def top_i_annotations(self, i):
        names = [usage.annotation_name for usage in self.usage_loader.load_all()]

        annotations = []
        values = []

        for name, number_of_usages in Counter(names).most_common():
            annotations += [name]
            values += [number_of_usages]

        pl = plt.barh(annotations[:i], values[:i], log=True)
        for bar in pl:
            width = bar.get_width()
            label_y = bar.get_y() + bar.get_height() / 2
            plt.text(width, label_y, s=f'{width}')

        plt.xlabel('number of usages', fontsize=15)
        plt.show()

    def top_i_annotations_by_target(self, i):
        for target, usages in self.usage_loader.usages_by_target.items():
            names = [usage.annotation_name for usage in usages]

            annotations = []
            values = []

            for name, number_of_usages in Counter(names).most_common():
                annotations += [name]
                values += [number_of_usages]

            pl = plt.barh(annotations[:i], values[:i], log=True)
            for bar in pl:
                width = bar.get_width()
                label_y = bar.get_y() + bar.get_height() / 2
                plt.text(width, label_y, s=f'{width}')

            plt.xlabel(target, fontsize=15)
            plt.show()
