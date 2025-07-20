import csv

inputfile= "higgs_output.txt"
ouputfile= "higgs_dataset.csv"

with open (inputfile,"r") as infile, open(ouputfile, "w", newline="") as outfile:
    writer=csv.writer(outfile)
    writer.writerow(["event", "lepton_index", "pid", "pT", "eta", "phi"])
    event_id=0
    lepton_index=0

    for line in infile:
        line = line.strip()
        if line == "----":
            event_id+=1
            lepton_index=0
        else:
            try:
                pid,pt,eta,phi = line.split()
                writer.writerow([event_id,lepton_index,pid,pt,eta,phi])
                lepton_index+=1
            except ValueError:
                continue