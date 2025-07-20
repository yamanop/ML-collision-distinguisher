#include "Pythia8/Pythia.h"
#include<fstream>

using namespace Pythia8;

int main() {
    Pythia pythia;
    pythia.readString("HiggsSM:all = on");  // Higgs production
    pythia.readString("Beams:eCM = 13000");
    pythia.init();

    std::ofstream outFile("higgs_data.csv");
    outFile << "event,particle_id,pt,eta,phi,label\n";

    int nEvents = 1000;
    for (int iEvent = 0; iEvent < nEvents; ++iEvent) {
        if (!pythia.next()) continue;

        for (int i = 0; i < pythia.event.size(); ++i) {
            if (!pythia.event[i].isFinal()) continue;

            outFile << iEvent << ","
                    << pythia.event[i].id() << ","
                    << pythia.event[i].pT() << ","
                    << pythia.event[i].eta() << ","
                    << pythia.event[i].phi() << ","
                    << "1\n";  // Label = 1 for Higgs
        }
    }

    outFile.close();
    pythia.stat();

    return 0;
}
