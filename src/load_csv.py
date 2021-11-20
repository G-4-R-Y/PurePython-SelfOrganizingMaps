class Data():
    def __init__(self, data_dir, header, index, normalize):
        """Defines a table structure, where lines
        contains features of each instance"""
        self.has_header = header
        self.has_index  = index
        self.num_cols   = 0
        self.index      = []
        self.header     = []
        self.features   = []
        self.full_table = []
        self.labels     = []
        self.load_csv(data_dir, header, index, normalize)

    def load_csv(self, data_dir, header, index, normalize):
        with open(data_dir, 'r') as f:
            for idx, line in enumerate(f.readlines()):
                row = self.read_row(line)
                if header == "True" and idx == 0:
                    self.header = row
                    continue
                if len(row) == 0 or row[0] == '': # empty (usually last) line
                    continue
                if (index == "True"):
                    self.index.append(row[0])
                    row = row[1:]
                row = [float(element) for element in row]
                self.features.append(row)
            self.num_cols = len(self.features[0])
        if normalize:
            self.normalize()
        self.test_data_empty()

    def normalize(self):
        means = []
        self.calc_means(means)
        stds  = []
        self.calc_stds(stds, means)
        self.norm_cols(means, stds)

    def calc_means(self, means):
        for j, column in enumerate(self.features[0]):  
            acc = 0
            for row in self.features:
                acc = acc + row[j]
            total = len(self.features)
            means.append(acc / total)

    def calc_stds(self, stds, means):
        for j, column in enumerate(self.features[0]):    
            deviations_acc = 0
            for row in self.features:
                deviations_acc = deviations_acc + abs(means[j] - row[j])
            total = len(self.features)
            stds.append(deviations_acc / total)

    def norm_cols(self, means, stds):
        norm_features = []
        for i, row in enumerate(self.features):
            norm_row = [(elem - means[j]) / stds[j] for j, elem in enumerate(row)]
            norm_features.append(norm_row)
        self.features = norm_features

    def test_data_empty(self):
        if self.num_cols == 0:
            print("Error - Empty dataset file")
            raise ValueError

    def read_row(self, line):
        row = line.split(',')
        row[-1] = row[-1].replace("\n", "")
        return row

    def save_full_table(self, header, index, output_dir, outputs):
        if header == "True":
            self.header.append("Cell_i")
            self.header.append("Cell_j\n")
            self.full_table.append(self.header)
        for idx, row in enumerate(self.features):
            if index == "True":
                row = [self.index[idx]] + row # concat lists
            row = row + outputs[idx]
            row = [str(elem) for elem in row]
            row[-1] += "\n"
            self.full_table.append(row)
        with open(output_dir, "w") as f:
            for row in self.full_table:
                f.write(','.join(row))