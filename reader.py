class Reader:
  @staticmethod
  def csv(input_file_name, should_skip_labels = False):
    f = open(input_file_name, "r")
    full_page = f.read()

    if full_page == '':
      return []

    rows = [row for row in full_page.split("\n") if row != '']
  
    if (should_skip_labels):
      rows = rows[1:]

    rows = [
      [float(c) for c in row.split(",")]
      for row in rows
    ]

    return rows
