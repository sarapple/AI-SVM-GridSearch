import sys
import numpy as np

from svm import SVM
from reporter import Reporter
from reader import Reader

def main():
  input_csv_file_name = sys.argv[1]
  output_csv_file_name = sys.argv[2]

  input_values = Reader.csv(input_csv_file_name, should_skip_labels = True)

  inputs = [x[0:2] for x in input_values]
  expected_classifiers = [int(x[-1]) for x in input_values]

  results = SVM.run(
    inputs = inputs,
    expected_classifiers = expected_classifiers,
  )
  results_strings = [",".join([str(item) for item in result]) for result in results]

  # write lines to output file
  Reporter.write_output(
    file_name = output_csv_file_name,
    content = "\n".join(results_strings),
    should_overwrite_file = True
  )

if __name__ == '__main__':
    main()
