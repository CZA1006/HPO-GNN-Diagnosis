import csv

target = "Developmental and epileptic encephalopathy 96"
codes = set()

with open("phenotype.hpoa") as f:
    # skip comment lines
    reader = csv.DictReader(
        (row for row in f if not row.startswith("#")),
        delimiter="\t"
    )
    # find the right column names
    # print(reader.fieldnames)  # uncomment to inspect header
    for row in reader:
        if row["disease_name"] == target:
            codes.add(row["hpo_id"])

print(sorted(codes))
